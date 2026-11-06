"""Wait for an actually idle supported GPU, then start exactly one remote tmux job."""
import argparse
import csv
import fcntl
import io
import json
from pathlib import Path
import shlex
import subprocess
import time


def idle_gpus(gpu_csv, process_csv):
    busy = {row[0].strip() for row in csv.reader(io.StringIO(process_csv)) if row}
    result=[]
    for row in csv.reader(io.StringIO(gpu_csv)):
        if len(row)!=4: raise ValueError('Unexpected GPU query columns')
        uuid,name,memory,util=map(str.strip,row)
        if uuid not in busy and float(memory)<256 and float(util)==0:
            result.append({'uuid':uuid,'name':name,'memory_mib':float(memory)})
    return result


def ssh(host, command):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
        '-o','ClearAllForwardings=yes',host,command],capture_output=True,text=True,timeout=45,check=True).stdout


def scan(host):
    command = 'nvidia-smi --query-gpu=uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits; printf "\\nPROCESSES\\n"; nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits'
    output = ssh(host,command) if host else subprocess.run(['bash','-c',command],capture_output=True,text=True,check=True,timeout=30).stdout
    gpu,proc=output.split('\nPROCESSES\n',1)
    return idle_gpus(gpu,proc)


def write(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--once',action='store_true');p.add_argument('--local',action='store_true')
    args=p.parse_args();cfg=json.loads(args.config.read_text());status=args.config.with_name('queue_status.json')
    with args.config.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if status.exists() and json.loads(status.read_text()).get('state') in ('launching','launched'):
            raise RuntimeError('This queue already launched or needs launch reconciliation')
        while True:
            cfg=json.loads(args.config.read_text())
            claim=Path(cfg['claim_directory']) if cfg.get('claim_directory') else None
            if claim and claim.exists():
                write(status,{'state':'claimed_elsewhere','claim_directory':str(claim)});return
            observations={};chosen=None
            for spec in cfg['hosts']:
                try:
                    if args.local:
                        missing=[name for name in spec.get('required_files',[]) if not Path(name).is_file()]
                        if missing:
                            observations[spec['host']]={'prepared':False,'missing_files':missing}
                            continue
                    scan_host=None if args.local else spec['host']
                    free=scan(scan_host)
                    if spec.get('allowed_gpu_uuids'):
                        free=[g for g in free if g['uuid'] in spec['allowed_gpu_uuids']]
                    observations[spec['host']]={'idle':free,'prepared':spec.get('prepared',False)}
                    if free and spec.get('prepared'):
                        # Require a second current observation; never use a stale snapshot.
                        time.sleep(5)
                        confirmed={g['uuid'] for g in scan(scan_host)}
                        candidate=next((g for g in free if g['uuid'] in confirmed),None)
                        if candidate:
                            chosen=(spec,candidate);break
                except Exception as exc: observations[spec['host']]={'error':str(exc)}
            now=time.strftime('%Y-%m-%d %H:%M:%S')
            if chosen:
                spec,gpu=chosen
                value={'state':'launching','checked_at':now,'host':spec['host'],'gpu':gpu,'session':cfg['session']}
                if claim:
                    claim.parent.mkdir(parents=True,exist_ok=True)
                    try:claim.mkdir()
                    except FileExistsError:
                        write(status,{'state':'claimed_elsewhere'});return
                    (claim/'owner.json').write_text(json.dumps(value,indent=2))
                write(status,value)
                command=['bash',spec['runner'],gpu['uuid'],cfg['run_tag'],spec['work_root'],spec['python'],spec['code_dir']]
                remote='tmux new-session -d -s '+shlex.quote(cfg['session'])+' '+shlex.quote(shlex.join(command))
                if args.local:
                    subprocess.run(['tmux','new-session','-d','-s',cfg['session'],shlex.join(command)],check=True)
                else:
                    ssh(spec['host'],remote)
                value['state']='launched';write(status,value);print(json.dumps(value),flush=True);return
            value={'state':'waiting_for_gpu','checked_at':now,'hosts':observations}
            write(status,value);print(json.dumps(value),flush=True)
            if args.once:return
            time.sleep(cfg.get('poll_seconds',60))

if __name__=='__main__':main()

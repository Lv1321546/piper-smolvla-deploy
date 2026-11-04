import unittest
from e14.collect_data import image_encoding


class ImageEncodingTests(unittest.TestCase):
    def test_default_png_unchanged(self):
        suffix,options=image_encoding({})
        self.assertEqual(suffix,'png')
        self.assertEqual(options,dict(format='PNG',compress_level=1))

    def test_jpeg_preserves_requested_quality_and_subsampling(self):
        suffix,options=image_encoding(dict(image_format='jpeg',jpeg_quality=90,jpeg_subsampling=0))
        self.assertEqual(suffix,'jpg')
        self.assertEqual(options,dict(format='JPEG',quality=90,subsampling=0))

    def test_invalid_codec_settings_rejected(self):
        for spec in [dict(image_format='webp'),dict(image_format='jpeg',jpeg_quality=True),
                     dict(image_format='jpeg',jpeg_quality=101),dict(image_format='jpeg',jpeg_subsampling=3)]:
            with self.assertRaises(ValueError):image_encoding(spec)


if __name__=='__main__':unittest.main()

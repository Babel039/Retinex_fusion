import tempfile
import unittest
from pathlib import Path


class FindImagePathsTest(unittest.TestCase):
    def test_returns_supported_images_in_natural_order_without_recursing(self):
        """防止文本文件和子目录图片被误送入当前批次。"""
        try:
            from Enhancement.my_prediction import find_image_paths
        except ModuleNotFoundError as error:
            if error.name == 'Enhancement.my_prediction':
                self.fail('Enhancement.my_prediction 尚未实现')
            raise

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / '10.jpg').touch()
            (input_dir / '2.PNG').touch()
            (input_dir / 'notes.txt').touch()
            nested_dir = input_dir / 'nested'
            nested_dir.mkdir()
            (nested_dir / '1.png').touch()

            paths = find_image_paths(input_dir)

            self.assertEqual(
                [path.name for path in paths],
                ['2.PNG', '10.jpg'],
            )


if __name__ == '__main__':
    unittest.main()

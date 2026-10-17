import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smolenv_local import LocalSandbox, SandboxConfig, _stage_file


class LocalSandboxStagingTests(unittest.TestCase):
    def make_sandbox(self, root: Path, files: list[str]) -> LocalSandbox:
        data_root = root / "data"
        work_root = root / "runs"
        return LocalSandbox(
            {"task_id": "staging", "bucket_prefix": "task", "files": files},
            data_root,
            work_root,
            SandboxConfig(backend="bwrap"),
        )

    def test_duplicate_same_source_basename_is_staged_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "data" / "task" / "insurance.csv"
            source.parent.mkdir(parents=True)
            source.write_text("policy_id,premium\n1,120\n")

            with patch("smolenv_local.platform.system", return_value="Linux"):
                sandbox = self.make_sandbox(root, ["insurance.csv", "insurance.csv"])

            staged = sandbox.input / "insurance.csv"
            self.assertTrue(staged.is_symlink())
            self.assertEqual(staged.resolve(), source.resolve())
            self.assertEqual(staged.read_text(), source.read_text())

    def test_distinct_sources_with_same_basename_still_raise(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prefix = root / "data" / "task"
            first = prefix / "first" / "insurance.csv"
            second = prefix / "second" / "insurance.csv"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text("source,first\n")
            second.write_text("source,second\n")

            with patch("smolenv_local.platform.system", return_value="Linux"):
                with self.assertRaises(FileExistsError):
                    self.make_sandbox(root, ["first/insurance.csv", "second/insurance.csv"])

            staged = next((root / "runs").glob("*/input/insurance.csv"))
            self.assertEqual(staged.resolve(), first.resolve())
            self.assertEqual(staged.read_text(), first.read_text())

    def test_existing_regular_file_is_not_replaced_by_a_symlink(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.csv"
            destination = root / "input.csv"
            source.write_text("source\n")
            destination.write_text("keep this file\n")

            with patch("smolenv_local.platform.system", return_value="Linux"):
                with self.assertRaises(FileExistsError):
                    _stage_file(source, destination, confined_writes=True)

            self.assertFalse(destination.is_symlink())
            self.assertEqual(destination.read_text(), "keep this file\n")


if __name__ == "__main__":
    unittest.main()

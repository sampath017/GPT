from pathlib import Path
from data_module import NamesDataModule


project_name = "Makemore"
block_size = 3
batch_size = 64

root_path = Path(".")
data_path = (root_path / "data/names.txt").absolute()
logs_path = (root_path / "data/logs").absolute()
logs_path.mkdir(parents=True, exist_ok=True)

dm = NamesDataModule(data_path, block_size=block_size, batch_size=batch_size)
dm.setup(stage="fit")

num_chars = dm.dataset.num_chars
start_char = dm.dataset.start_char
end_char = dm.dataset.end_char
ctoi = dm.dataset.ctoi
itoc = dm.dataset.itoc

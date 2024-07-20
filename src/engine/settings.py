from pathlib import Path
from .data_module import ShakespearDataModule


project_name = "GPT"
context_size = 8
batch_size = 64
epochs = 20
limit_train_batches = None
limit_val_batches = None
debug = False

root_path = Path(__file__).parent.parent.parent
data_path = (root_path / "data/tiny_shakespeare.txt").absolute()
logs_path = (root_path / "data/logs").absolute()
logs_path.mkdir(parents=True, exist_ok=True)

dm = ShakespearDataModule(
    data_path, context_size=context_size, batch_size=batch_size)
dm.setup(stage="fit")

num_chars = dm.dataset.num_chars
decode = dm.dataset.decode
itoc = dm.dataset.itoc
ctoi = dm.dataset.ctoi

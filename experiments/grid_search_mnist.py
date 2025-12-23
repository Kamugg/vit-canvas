import os
import time
from pathlib import Path
import torch
from torch.nn import Module
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Normalize, RandomAffine, ToImage, ToDtype
from torchvision.datasets import MNIST
import json
from sklearn.model_selection import ParameterGrid
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall
import onnxruntime as ort
from custom_modules.vision_transformer import VisionTransformer
from copy import deepcopy

INPUT_DIM = 28
PATCH_SIZE = 4
NUM_CHANNELS = 1
NUM_CLASSES = 10
MAX_EPOCHS = 100
PATIENCE = 3
PATIENCE_DELTA = 0.01
BATCH_SIZE = 1024
LR = 1e-3
DEVICE = 'cuda'
TIME_SAMPLES = 500
SEED = 42

criterion = CrossEntropyLoss()
accuracy = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(DEVICE)
precision = Precision(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(DEVICE)
recall = Recall(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(DEVICE)
torch.manual_seed(SEED)

@torch.no_grad()
def eval_model(vit: Module, val_loader: DataLoader) -> dict:
    """
    Runs the evaluation loop.
    :param vit: VIT model
    :param val_loader: Validation dataloader
    :return: Dictionary of evaluation metrics (loss, accuracy, precision, recall, f1)
    """

    # Init
    vit.eval()
    accuracy.reset()
    precision.reset()
    recall.reset()
    pbar = tqdm(val_loader)
    pbar.set_description('Validation')
    val_loss = 0.0

    # Run over the validation set and update metrics
    for i, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        out = vit(images)['scores']
        val_loss += criterion(out, labels).item()
        pbar.set_postfix(loss=val_loss / (i+1))
        accuracy.update(out, labels)
        precision.update(out, labels)
        recall.update(out, labels)

    # Compute metrics and return
    results = {
        'val_loss': val_loss / len(val_loader),
        'val_accuracy': accuracy.compute(),
        'val_precision': precision.compute(),
        'val_recall': recall.compute()
    }
    results['val_f1'] = 2 * results['val_precision'] * results['val_recall'] / (results['val_precision'] + results['val_recall'])
    return results


def train_loop(vit: Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
    """
    Runs the training loop and the evaluation loop.
    :param vit: VIT model
    :param train_loader: Train dataloader
    :param val_loader: Validation dataloader
    :return: None
    """

    # Init
    vit.train()
    optimizer = AdamW(vit.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, MAX_EPOCHS)
    patience_counter = PATIENCE
    best_val_loss = torch.inf
    best_params = None

    # Run up to max epochs
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        pbar = tqdm(train_loader)
        pbar.set_description(f'Epoch: {epoch+1}/{MAX_EPOCHS}')
        for i, (images, labels) in enumerate(pbar):
            optimizer.zero_grad()
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = vit(images)['scores']
            loss = criterion(logits, labels)
            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss / (i+1))
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Eval

        results = eval_model(vit, val_loader)
        results['train_loss'] = epoch_loss / len(train_loader)
        wandb.log(results)

        # Save best state
        if best_val_loss > results['val_loss']:
            print('Validation loss improved. Saving...')
            best_params = deepcopy(vit.state_dict())

        # Patience check
        if (best_val_loss < results['val_loss']) or ((best_val_loss - results['val_loss']) < PATIENCE_DELTA):
            print(f'Patience proc!\nBest val loss: {best_val_loss}\nVal loss this epoch: {results["val_loss"]}')
            patience_counter -= 1
            if patience_counter == 0:
                print('Patience expired.')
                break
        else:
            if patience_counter != PATIENCE:
                print('Patience reset.')
                patience_counter = PATIENCE
            best_val_loss = results['val_loss']

    if best_params is not None:
        vit.load_state_dict(best_params)

def time_evaluation(vit: Module, run_id: int) -> None:
    """
    Evaluates the model latency using standard torch and onnx
    :param vit: The VIT model
    :param run_id: Run number
    :return: None
    """

    vit.eval()
    vit = vit.to('cpu')

    # Dummy input
    dummy = torch.randn(1, 1, 28, 28)

    # Warmup
    for _ in range(20): _ = vit(dummy)

    # Evaluate torch inference runtime
    pbar = tqdm(range(TIME_SAMPLES))
    pbar.set_description('Evaluating torch inference runtime')
    t0 = time.perf_counter()
    for _ in pbar: _ = vit(dummy)
    t1 = time.perf_counter()
    torch_latency = ((t1 - t0) / TIME_SAMPLES) * 1000

    # Create paths for the onnx files
    script_dir = Path(__file__).parent.resolve()
    chkps_dir = script_dir / 'chkps'
    chkps_dir.mkdir(parents=True, exist_ok=True)
    onnx_filename =  chkps_dir / f"microvit_run_{run_id}.onnx"
    onnx_data_filename = chkps_dir / f"microvit_run_{run_id}.onnx.data"

    # Set outputs names and dynamic axes
    output_names = [f'cls_att_tblock_{i}' for i in range(vit.n_layers)] + ['scores']
    dynamic_axes = {'input': {0: 'batch_size'}}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}

    torch.onnx.export(
        vit,
        dummy,
        onnx_filename,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Benchmark times for ONNX
    ort_session = ort.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy.numpy()}
    for _ in range(20): ort_session.run(None, ort_inputs)
    pbar = tqdm(range(TIME_SAMPLES))
    pbar.set_description('Evaluating ONNX inference runtime')
    t0 = time.perf_counter()
    for _ in pbar: ort_session.run(None, ort_inputs)
    t1 = time.perf_counter()
    onnx_latency = ((t1 - t0) / TIME_SAMPLES) * 1000

    # Measure total file size
    size_kb = os.path.getsize(onnx_filename) / 1024
    if onnx_data_filename.exists():
        size_kb += os.path.getsize(onnx_data_filename) / 1024

    wandb.log({
        "torch_inference_ms": torch_latency,
        "onnx_inference_ms": onnx_latency,
        "latency_reduction_pct": (torch_latency - onnx_latency) / torch_latency * 100,
        "model_size_kb": size_kb
    })

    # Save artifacts to W&B
    wandb.save(onnx_filename)

    if onnx_data_filename.exists():
        wandb.save(onnx_data_filename)


def main():
    """
    Runs the grid-search experiment using the parameters stored in param_grid.json
    :return: None
    """

    # Loading parameter grid
    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent
    print('Loading parameter grid...')
    with open(script_dir / 'param_grid.json', 'r') as f:
        grid_data = json.load(f)
    pgrid = ParameterGrid(grid_data)
    print(f"Total combinations: {len(pgrid)}")

    # Preparing datasets
    print('Loading MNIST dataset...')
    train_split = MNIST(root_dir / 'datasets',
                        download=True,
                        train=True,
                        transform=Compose([
                            ToImage(),
                            ToDtype(torch.float32, scale=True),
                            RandomAffine(
                                degrees=5,  # Rotation
                                translate=(0.3, 0.3),  # Shift
                                scale=(0.7, 1.1)  # Zoom
                            ),
                            Normalize((0.1307,), (0.3081,)),
                        ]))
    val_split = MNIST(root_dir / 'datasets',
                      download=True,
                      train=False,
                      transform=Compose([
                            ToImage(),
                            ToDtype(torch.float32, scale=True),
                            Normalize((0.1307,), (0.3081,)),
                        ]))

    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False)

    # Start grid search
    for run_id, params in enumerate(pgrid):

        # Also init W&B logging
        wandb.init(project='MNIST-vit-gridsearch', config=params, name=f'run_{run_id}')
        vit = VisionTransformer(input_dim=INPUT_DIM,
                                patch_size=PATCH_SIZE,
                                num_channels=NUM_CHANNELS,
                                num_classes=NUM_CLASSES,
                                **params)
        vit = vit.to(device=DEVICE)
        train_loop(vit, train_loader, val_loader)
        time_evaluation(vit, run_id)
        wandb.finish()


if __name__ == "__main__":
    main()
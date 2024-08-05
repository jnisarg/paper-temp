import os
import warnings
from io import BytesIO

import onnx
import onnxsim
import torch
from rich.console import Console

from model import Model

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
console = Console()


class OnnxModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        classifier, _, centerness, regression = self.model(x)
        return classifier.argmax(dim=1, keepdims=True), centerness, regression


def export():
    model = Model()

    work_dir = "exp"
    # work_dir = "exp/paper_exp1_0.1_bbox_loss/"
    # model_path = work_dir + "checkpoints/model_best.pth"
    # snapshot = torch.load(model_path)

    # console.log(f"Loading model from {model_path} at epoch {snapshot['epoch']}\n")

    # state_dict = snapshot["model_state_dict"]
    # # state_dict = {
    # #     k.replace("module.model.", ""): v
    # #     for k, v in state_dict.items()
    # #     if k.startswith("module.model.")
    # # }
    # state_dict = {
    #     k.replace("model.", ""): v
    #     for k, v in state_dict.items()
    #     if k.startswith("model.")
    # }

    # model.load_state_dict(state_dict, strict=False)

    model = OnnxModel(model)
    model.eval()

    os.makedirs(os.path.join(work_dir, "onnx_files"), exist_ok=True)

    sample = torch.randn(1, 3, 384, 768)
    input_names, output_names = ["input_tensor"], [
        "classification",
        "centerness",
        "regression",
    ]

    onnx_path = os.path.join(work_dir, "onnx_files", "model.onnx")
    with BytesIO() as f:
        torch.onnx.export(
            model,
            sample,
            f,
            verbose=False,
            opset_version=11,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
        )
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)

        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "simplify check failed"
        except Exception as error:
            print(f"Simplifier failure: {error}")
    onnx.save(onnx_model, onnx_path)

    console.log(f"Onnx saved at: {onnx_path}")
    console.log(
        f"\nTo visualize the graph, you can run the following command: ```netron {onnx_path}```\n"
    )


if __name__ == "__main__":
    export()

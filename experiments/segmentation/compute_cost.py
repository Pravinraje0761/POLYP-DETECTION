
import os
import time
import torch
import segmentation_models_pytorch as smp
from polypgate.models.adenseunet import ADenseUNet_Boosted
from thop import profile


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 320
BEST_UNETPP = "checkpoints/best_pretrained_unetpp.pth"
BEST_ADENSE = "checkpoints/best_adenseunet_combined_honest.pth"
OUTPUT_PATH = "flops_latency_results.json"


def count_parameters(model):
    encoder_params = 0
    decoder_params = 0
    if hasattr(model, "init_conv"):  # ADenseUNet
        encoder = [model.init_conv, model.dense1, model.trans1, model.dense2, model.trans2, model.dense3, model.trans3, model.bottleneck]
        decoder = [model.up3, model.att3, model.dec3, model.up2, model.att2, model.dec2, model.up1, model.att1, model.dec1, model.up_final, model.final_conv]
        for mod in encoder:
            encoder_params += sum(p.numel() for p in mod.parameters())
        for mod in decoder:
            decoder_params += sum(p.numel() for p in mod.parameters())
    elif hasattr(model, "encoder") and hasattr(model, "decoder"):  # SMP
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        if hasattr(model, "segmentation_head"):
            decoder_params += sum(p.numel() for p in model.segmentation_head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    return encoder_params, decoder_params, total_params


def measure_latency(model, input_tensor, num_runs=100):
    model.eval()
    with torch.no_grad():
        # warm-up runs
        for _ in range(10):
            _ = model(input_tensor)

        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_tensor)
        total_time = time.time() - start_time
        avg_latency = total_time / num_runs * 1000  # in milliseconds
    return avg_latency


def measure_flops(model, input_tensor):
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops, params


def main():
    input_tensor = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    results = {}

    # Load and evaluate Pretrained Unet++
    print("=== Evaluating Pretrained Unet++ ===")
    ckpt = torch.load(BEST_UNETPP, map_location=DEVICE, weights_only=False)
    model_pp = smp.UnetPlusPlus(encoder_name=ckpt["encoder"], encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    model_pp.load_state_dict(ckpt["model_state_dict"])
    enc_pp, dec_pp, tot_pp = count_parameters(model_pp)
    flops_pp, _ = measure_flops(model_pp, input_tensor)
    latency_pp = measure_latency(model_pp, input_tensor)
    results["Pretrained_UnetPP"] = {
        "encoder_params": enc_pp, "decoder_params": dec_pp, "total_params": tot_pp,
        "flops": flops_pp, "latency_ms": latency_pp
    }
    print(f"Params: {tot_pp/1e6:.2f}M, FLOPs: {flops_pp/1e9:.2f}G, Latency: {latency_pp:.2f} ms")

    # Load and evaluate ADenseUNet
    print("\n=== Evaluating ADenseUNet_combined_honest ===")
    ckpt = torch.load(BEST_ADENSE, map_location=DEVICE, weights_only=False)
    model_ad = ADenseUNet_Boosted(in_channels=3, out_channels=1).to(DEVICE)
    state_dict = ckpt["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("final_up."):
            new_state_dict[k.replace("final_up.", "up_final.")] = v
        elif k.startswith("up_final."):
            new_state_dict[k] = v
        else:
            new_state_dict[k] = v
    model_ad.load_state_dict(new_state_dict, strict=True)
    enc_ad, dec_ad, tot_ad = count_parameters(model_ad)
    flops_ad, _ = measure_flops(model_ad, input_tensor)
    latency_ad = measure_latency(model_ad, input_tensor)
    results["ADenseUNet_combined_honest"] = {
        "encoder_params": enc_ad, "decoder_params": dec_ad, "total_params": tot_ad,
        "flops": flops_ad, "latency_ms": latency_ad
    }
    print(f"Params: {tot_ad/1e6:.2f}M, FLOPs: {flops_ad/1e9:.2f}G, Latency: {latency_ad:.2f} ms")

    # Save results
    import json
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_PATH}!")


if __name__ == "__main__":
    main()

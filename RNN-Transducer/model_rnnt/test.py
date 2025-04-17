import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Transducer
from encoder import BaseEncoder
from decoder import BaseDecoder
import yaml

# ========== Create Model From Config ==========
def create_model(config):
    enc_cfg = config['model']['enc']
    dec_cfg = config['model']['dec']
    joint_cfg = config['model']['joint']

    encoder = BaseEncoder(
        input_size=enc_cfg['input_size'],
        hidden_size=enc_cfg['hidden_size'],
        output_size=enc_cfg['output_size'],
        n_layers=enc_cfg['n_layers'],
        bidirectional=enc_cfg.get('bidirectional', False)
    )

    decoder = BaseDecoder(
        embedding_size=dec_cfg['embedding_size'],
        hidden_size=dec_cfg['hidden_size'],
        vocab_size=config['model']['vocab_size'],
        output_size=dec_cfg['output_size'],
        n_layers=dec_cfg['n_layers']
    )

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        input_size=joint_cfg['input_size'],
        inner_dim=joint_cfg['inner_dim'],
        vocab_size=config['model']['vocab_size']
    )

    return model

# ========== Test Logic ==========
if __name__ == "__main__":
    # Load config
    with open("/data/npl/Speech2Text/RNN-Transducer/label,csv/RNN-T_mobile_2.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = create_model(config)

    # ===== Dummy Data Matching Config =====
    B = 2  # batch size
    T = 50  # time steps
    U = 20  # target length

    input_dim = config['model']['enc']['input_size']
    proj_dim = config['model']['enc']['output_size']
    vocab_size = config['model']['vocab_size']

    inputs = torch.randn(B, T, input_dim)
    input_lengths = torch.tensor([T] * B)
    targets = torch.randint(1, vocab_size, (B, U))
    target_lengths = torch.tensor([U] * B)

    # ===== Test JointNet =====
    print("\n--- Testing JointNet shape ---")
    joint = model.joint
    e = torch.randn(B, T, U, proj_dim)
    d = torch.randn(B, T, U, proj_dim)
    j_out = joint(e, d)
    assert j_out.shape == (B, T, U, vocab_size), f"Expected {(B, T, U, vocab_size)} but got {j_out.shape}"
    print("✅ Passed JointNet test.")

    # ===== Test Forward =====
    print("\n--- Testing forward() ---")
    model.train()
    logits, loss = model(inputs, input_lengths, targets, target_lengths)
    print("✅ Forward output logits shape:", logits.shape)
    print("✅ Forward loss:", loss.item())

    # ===== Test Recognize =====
    print("\n--- Testing recognize() ---")
    model.eval()
    with torch.no_grad():
        results = model.recognize(inputs, input_lengths)
        print("✅ Recognize output:", results)

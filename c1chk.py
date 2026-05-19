import torch
for ds in ['TLD', 'AFHQ']:
    ckpt = torch.load(f'E:\\C1_TRAINED\\{ds}\\ckpts\\c1_best.pth', map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    # Check critical keys
    critical = ['wm_affine', 'gate_strength', 'destructive_strength',
                'base.layer2.0.downsample.1.k']
    print(f'=== {ds} C1 ===')
    print(f'  best_val_acc: {ckpt.get(\"best_val_acc\", \"?\")}')
    print(f'  num_classes:  {ckpt.get(\"num_classes\", \"?\")}')
    for k in critical:
    print(f'  {\"✓\" if k in sd else \"✗\"} {k}')
    "
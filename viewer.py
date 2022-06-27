if __name__ == '__main__':
    import plenoctree as plnoct
    import argparse
    import numpy as np
    import os

    OUTPUT_DIR_ROOT = './output'
    if not os.path.exists(OUTPUT_DIR_ROOT):
        print(f'create output directory {OUTPUT_DIR_ROOT}')
        os.makedirs(OUTPUT_DIR_ROOT)

    parser = argparse.ArgumentParser()
    parser.add_argument('tree_npz', help='path to the plenoctree npz file')
    parser.add_argument('-p', '--palette', help='path to the palette npz file')
    args = parser.parse_args()

    tree = plnoct.PlenOctree(args.tree_npz)

    if args.palette is not None:
        palette_rgb = np.load(args.palette)['palette']
        print(f'palette shape: {palette_rgb.shape}')
        tree.show(palette=palette_rgb, output_dir=OUTPUT_DIR_ROOT)
    else:
        tree.show(output_dir=OUTPUT_DIR_ROOT)

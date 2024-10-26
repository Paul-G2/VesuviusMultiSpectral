import sys
import datetime
import argparse
from layers_from_ppm import LayersFromPPM


if __name__ == '__main__':

    # Build a parser for the command-line args
    parser = argparse.ArgumentParser(prog='layers_from_ppm', description='Creates a segment from a .ppm file.')
    requiredNamedArgs = parser.add_argument_group('required named arguments')
    requiredNamedArgs.add_argument('-v', '--volpkg', help='VolumePkg path', required=True)
    requiredNamedArgs.add_argument('-p', '--ppm', help='Input PPM file', required=True)
    requiredNamedArgs.add_argument('--volume', help='Volume to use for texturing', required=True)
    parser.add_argument('--transform', default = None, help='Path to a Transform3D .json file')
    parser.add_argument('--invert-transform', default=False, action=argparse.BooleanOptionalAction, help='When provided, invert the transform')
    requiredNamedArgs.add_argument('-o', '--output-dir', help='Output directory for layer images', required=True)
    parser.add_argument('-f', '--image-format', choices={'png', 'tif', 'tiff'}, default='png', help='Image format for layer images. Default: png')
    parser.add_argument('--compression', type=int, default=0, choices=range(0,10), metavar="[0-9]", help='Compression level for png output images')
    parser.add_argument('-r', '--radius', type=float, default=None, help='Search radius. Defaults to value calculated from estimated layer thickness')
    parser.add_argument('-i', '--interval', type=float, default=1.0, help='Sampling interval')
    parser.add_argument('-d', '--direction', type=int, choices={-1, 0, 1}, default=0, help='Sample Direction: -1:Negative, 0:Both, 1:Positive')
    parser.add_argument('--cache-memory-limit', default = None, help='Maximum size of the slice cache in bytes. Accepts the suffixes: (K|M|G|T)(B). ' +
            'Default: 50%% of the total system memory.')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # Configure a worker instance
    lfp = LayersFromPPM()
    lfp.volpkg_dir = args.volpkg
    lfp.ppm_filepath = args.ppm
    lfp.volume_name = args.volume
    lfp.transform_filepath = args.transform
    lfp.invert_transform = args.invert_transform
    lfp.output_dir = args.output_dir
    lfp.image_format = args.image_format
    lfp.compression = args.compression
    lfp.radius = args.radius
    lfp.sampling_interval = args.interval
    lfp.sampling_direction = args.direction
    lfp.cache_mem_limit = None if args.cache_memory_limit is None else \
            LayersFromPPM.parse_memsize_string(args.cache_memory_limit)
    lfp.progress_handler = LayersFromPPM.default_progress_handler

    # Compute the layers
    print('Started at ' + datetime.datetime.now().strftime("%H:%M:%S"))
    lfp.run()
    print('\nFinished at ' + datetime.datetime.now().strftime("%H:%M:%S"))
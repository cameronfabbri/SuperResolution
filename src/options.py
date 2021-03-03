"""
"""


def get_data_args(parser):
    parser.add_argument(
        '--data_dir', required=False, default='data', type=str,
        help='')
    parser.add_argument(
        '--cache_size', required=False, default=20, type=int,
        help='')
    parser.add_argument(
        '--num_workers', required=False, default=0, type=int,
        help='')
    parser.add_argument(
        '--patch_size', required=False, default=128, type=int,
        help='')
    parser.add_argument(
        '--num_ds', required=False, default=2, type=int,
        help='')
    return parser


def get_network_args(parser):
    parser.add_argument(
        '--batch_size', required=False, default=8, type=int,
        help='')
    parser.add_argument(
        '--block_type', required=False, default='sisr', type=str,
        help='Type of residual block',
        choices=['sisr', 'rrdb'])
    parser.add_argument(
        '--num_blocks', required=False, default=16, type=int,
        help='Number of residual blocks to use')
    return parser


def get_loss_args(parser):
    parser.add_argument(
        '--lambda_l1', required=False, default=100., type=float,
        help='Scalar for L1 value')
    parser.add_argument(
        '--lr_g', required=False, default=1e-3, type=float,
        help='')
    return parser


def parse(parser, argv):
    parser = get_data_args(parser)
    parser = get_network_args(parser)
    parser = get_loss_args(parser)

    return parser.parse_args(argv[1:])

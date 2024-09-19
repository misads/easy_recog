# python 3.5, pytorch 1.14
from network import get_model
from utils import raise_exception, Context

if __name__ == '__main__':
    from options import base_options, parse_args, set_config
    from network import get_model
    from misc_utils import get_file_name

    opt = base_options()
    opt.add_argument('--load', type=str, default=None, help='指定载入checkpoint的路径')
    opt.add_argument('--save', type=str, default=None, help='导出的onnx路径')
    opt = parse_args(opt)
    config = set_config(opt)

    if opt.save is None:
        opt.save = get_file_name(opt.config) + '.onnx'

    if not opt.load:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')

    with Context('init model'):
        model = get_model(opt, config)
        model = model.cpu()

        load_epoch = model.load(opt.load)
        if load_epoch is not None:
            opt.which_epoch = load_epoch

        model.eval()

    model.export_onnx(opt.save)
    model.check_onnx(opt.save)
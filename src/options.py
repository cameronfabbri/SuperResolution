import argparse


def print_argsions(self, args):
    message = '\n----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = self.parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

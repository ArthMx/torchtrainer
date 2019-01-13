import numpy as np
import collections
import time
import sys

class Progbar(object):
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        val_target: Total number of validation steps expected, None if not validation.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, val_target=None, width=30, verbose=1, interval=0.05):
        self.target = target
        self.val_target = val_target
        self.width = width
        self.verbose = verbose
        self.interval = interval

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self.current = 0
        if self.val_target:
            self.val_current = 0
            self.val_step = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0
        self.validating = False
        
        self.train_losses = []
        self.val_losses = []
        

    def update(self, values=None, validating=False):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
        """
        values = values or []
        for k, v in values:
            if k not in self._values:
                self._values[k] = AverageMeter()
            
            self._values[k].update(v)
        
        if not validating:
            self.current += 1
        else:
            self.val_current += 1
            if self.val_step == 0:
                self.val_step = 1
                self.current = 1
                self.target = self.val_target
            else:
                self.current += 1
            

        now = time.time()
        if self.verbose == 1:
            if now - self._last_update < self.interval and \
            self.current != self.target and \
            self.val_current != self.val_target:
                return

            line = "\r"
            
            if self.current < self.target:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % self.current
                prog = float(self.current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if self.current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
                    
                line += bar


            running_time = now - self._start
            time_per_unit = running_time / self.current
                
            if self.current < self.target:
                eta = time_per_unit * (self.target - self.current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                info = ""
                if running_time > 3600:
                    info += ('%d:%02d:%02d' % (running_time // 3600, 
                                               (running_time % 3600) // 60, 
                                               running_time % 60))
                elif running_time > 60:
                    info += '%d:%02d' % (running_time // 60, running_time % 60)
                elif running_time >= 1:
                    info += '%.3fs' % running_time
                elif  running_time >= 1e3:
                    info += '%dms' % int(running_time  * 1e3)
                elif running_time >= 1e6:
                    info += '%dus' % int(running_time * 1e6)
            
            for k in self._values:
                info += ' - %s:' % k
                avg = self._values[k].average()
                if abs(avg) > 1e-3:
                    info += ' %.4f' % avg
                else:
                    info += ' %.4e' % avg
            
            info += ' ' * 50
            if (self.val_target is None) and (self.current == self.target) or \
                (self.val_target is not None) and (self.val_current == self.val_target):
                    info += '\n'
            
            line += info
            sys.stdout.write(line)
            sys.stdout.flush()
            
            
        elif self.verbose == 2:
            if (self.val_target is None and self.current == self.target) or \
                    (self.val_current == self.val_target):
                info = ""
                running_time = now - self._start
                if running_time > 3600:
                    info += ('%d:%02d:%02d' % (running_time // 3600, 
                                               (running_time % 3600) // 60, 
                                               running_time % 60))
                elif running_time > 60:
                    info += '%d:%02d' % (running_time // 60, running_time % 60)
                elif running_time >= 1:
                    info += '%.3fs' % running_time
                elif  running_time >= 1e3:
                    info += '%dms' % int(running_time  * 1e3)
                elif running_time >= 1e6:
                    info += '%dus' % int(running_time * 1e6)
            
                for k in self._values:
                    info += ' - %s:' % k
                    avg = self._values[k].average()
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
        
                info += '\n'
                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now
        
class AverageMeter(object):
    """Sum values to compute the mean."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        
    def update(self, val):
        self.count += 1
        self.sum += val
        
    def average(self):
        return self.sum / self.count
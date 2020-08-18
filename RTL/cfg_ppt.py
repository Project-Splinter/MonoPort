import os
import cv2


_DATA_FOLDER = os.path.join(
    os.path.dirname(__file__), '../data/RTL/')


# under _DATA_FOLDER
# _FILES = [
#     './board/blackboard.jpeg',
#     './board/N6ZUU.png']

# _FILES = [os.path.join('Zeng_Defense', fname) 
#     for fname in sorted(os.listdir(os.path.join(_DATA_FOLDER, 'Zeng_Defense'))) if 'Wide' in fname]

_FILES = [os.path.join('Zeng_Defense', f'Wide{i}.PNG') for i in range(1, 44)]


def _load(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class CfgPPT(object):
    def __init__(self):
        self.cur_idx = 0 # current slides index
        self.cur_ts = 0 # current timestamp for this page
        self.slides = [_load(os.path.join(_DATA_FOLDER, fname)) for fname in _FILES]
        
        self.current_yaw = 0
        self.current_pitch = 0
        self.direction = 1

    def next(self):
        self.cur_idx = min(self.cur_idx + 1, len(self.slides) - 1)
        print ('next slides', self.cur_idx)
        return self.current()

    def last(self):
        self.cur_idx = max(self.cur_idx - 1, 0)
        print ('last slides', self.cur_idx)
        return self.current()

    def current(self):
        self.cur_ts = 0 # reset timestamp to zero
        return self.slides[self.cur_idx]

    def get_current_yaw_pitch(self):
        self.cur_ts += 1
        eps = 1 / 4
        slower_f = 1/2
        p_limit = 16
        headline_slides = [1, 6, 15, 32, 41, 42]
        last_silde = 42
        yaw_limit = 45
        if self.cur_idx == 0:
            self.current_yaw = 0
            self.current_pitch = 0
        if self.cur_idx > 4:
            eps = 1
            slower_f = 1 / 8
            p_limit = 8
            self.current_pitch = 0
            yaw_limit = 20
        if self.cur_idx == last_silde:
            eps = 1
            slower_f = 1/2
            p_limit = 45
            self.current_pitch = 0
            yaw_limit = 45
        if self.cur_idx in headline_slides: # first slides
            #gradually go to yaw=45
            if self.current_yaw < 45:
                self.current_yaw += eps
            # gradually go to pitch=0
            if self.current_pitch > eps:
                self.current_pitch -= eps
            if self.current_pitch < -eps:
                self.current_pitch += eps
            if abs(self.current_pitch) <= eps:
                self.current_pitch = 0
        else:
            # need to be implemented for other slides
            if self.current_yaw > yaw_limit:
                    self.current_yaw -= eps
            if self.current_pitch > p_limit:
                self.direction = -1
            elif self.current_pitch < -p_limit:
                self.direction = 1
            self.current_pitch += self.direction * eps * slower_f
        return self.current_yaw, self.current_pitch
            


cfg_ppt = CfgPPT()
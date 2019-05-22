import numpy as np

class norm_monitor(object):

  def __init__(self, digits=1, layers=7, rel_res=1E-5, interval=1, stride=1, layermap=[]):
    self.res_max = 1E10
    self.digits = digits
    self.layers = layers
    self.REL_RES = rel_res
    self.interval = interval
    self.stride = stride

    if len(layermap) == 0:
      self.layermap = [1 for i in range(layers)]
    else:
      self.layermap = layermap
    self.layerinfo = [[self.digits, self.res_max, 0] for i in range(layers)]

  def adjust_digits(self, norm_list):
    wl = np.median(norm_list, axis=0)
    for i, lm in enumerate(self.layermap):
      if lm == 0:
        continue

      if self.layerinfo[i][1] != self.res_max:
        residual = wl[i] - self.layerinfo[i][1]
        rel_res = abs(residual/self.layerinfo[i][1])
        if (rel_res-self.REL_RES) > 0 or self.layerinfo[i][1] is np.inf:
          self.layerinfo[i][1] = wl[i]
          self.layerinfo[i][2] = 0
        else:
          self.layerinfo[i][2] += 1

        if self.layerinfo[i][2] == self.interval:
          if self.layerinfo[i][0] < 32:
            self.layerinfo[i][0] += self.stride
          self.layerinfo[i][1] = self.res_max
          self.layerinfo[i][2] = 0
      else:
        self.layerinfo[i][1] = wl[i]
        self.layerinfo[i][2] += 1

  def get_layerinfo(self):
      return self.layerinfo

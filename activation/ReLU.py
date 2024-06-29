import numpy as np

class RelU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


import numpy as np
import matplotlib.pyplot as plt


class generate_one_f:
    # コンストラクタで一気に1/f製造する。
    def __init__(self, duration):
        self.random_num = []
        """
        while duration > 0:
            self.random_num.append(random.random() - 0.5)
            duration -= 1
            
            高速化する。
        """

        self.random_num = np.random.rand(duration) - 0.5

        # print(self.random_num)
        # print(len(self.random_num))
        self.fft_result = np.fft.fft(self.random_num)
        self.fft_result_Numpy = np.array(self.fft_result)
        self.random_sqrt = []
        """
        for self.ind, self.num in enumerate(self.fft_result_Numpy):
            if self.ind == 0:
                self.random_sqrt.append(self.fft_result_Numpy[0])
            else:
                self.random_sqrt.append(self.num / np.sqrt(self.ind))

                これも高速化する。
        """
        self.freqs = np.arange(len(self.fft_result_Numpy))
        self.freqs[0] = 1
        self.random_sqrt = self.fft_result_Numpy / np.sqrt(self.freqs)

        self.ifft_result = np.array(np.fft.ifft(self.random_sqrt))
        self.ifft_real_result = self.ifft_result.real * 100 + 1

        print(self.ifft_real_result)

    def one_f_visualize(self):
        fig, ax = plt.subplots()
        self.y = self.ifft_real_result
        ax.plot(self.y)
        plt.show()


if __name__ == "__main__":  # テスト
    a = generate_one_f(100)
    a.one_f_visualize()

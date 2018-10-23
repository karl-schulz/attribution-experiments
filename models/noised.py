class Noised(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        print("hooking {}".format(type(original).__name__))

    def show_actmap(self, x):
        while len(x.shape) < 4:
            print(x.shape)
            x = x.unsqueeze(0)
        hmap = x.detach().cpu().numpy()
        hmap = hmap.mean(axis=(0, 1))
        print(hmap.shape)
        plt.figure()
        plt.imshow(hmap, cmap="Greys")

    def show_hist(self, x):
        acts = x.detach().cpu().numpy()
        acts = acts.flatten()
        print(acts.mean())
        print(acts.std())

        f = plt.figure(figsize=(20, 10))
        plt.hist(acts, bins=200, density=True)

        mu, std = norm.fit(acts)
        xmin, xmax = min(acts), max(acts)
        plt_x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(plt_x, mu, std)
        plt.plot(plt_x, p, 'k', linewidth=2)

        f = plt.figure()

    def forward(self, x):
        x = self.original.forward(x)
        self.show_hist(x[0])
        return x
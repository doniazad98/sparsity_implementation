import matplotlib.pyplot as plt
#plt.plot([1, 2, 3, 4])
import numpy as np


k=[10, 20, 30, 40,50]
MAEtopsis2AlphaFull=[0.682385113559013, 0.6928144611259032, 0.7032611822226619, 0.7113975001489381,0.7172221662997247]
MAEknn2AlphaFull=[0.6982859509740496,0.7045871902004741,0.712131101322214,0.7177340363595236,0.7221699299632558]
MAEknn2AlphaFull_IGN_SIM=[0.7764878700603417,0.7592510257668703,0.7525320083499317,0.7492415956026719,0.7473038466103293]

"""
plt.ylabel('MAE')
plt.figure(figsize=(9, 3))
#plt.subplot(131)
plt.plot(k,MAEtopsis2AlphaFull, )
plt.plot(k,MAEknn2AlphaFull,'k' )


#plt.subplot(132)
#plt.plot(k,MAEtopsis2AlphaFull, )
#plt.plot(k,MAEknn2AlphaFull_IGN_SIM,'r--')
plt.suptitle('MAE')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.show()
"""
x = np.random.randint(low=1, high=11, size=50)
y = x + np.random.randint(1, 5, size=x.size)
data = np.column_stack((x, y))

fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3,figsize=(15, 4))

#ax1.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')
ax1.plot(k,MAEtopsis2AlphaFull,label='Topsis' )
ax1.plot(k,MAEknn2AlphaFull,'k',label='KNN simple' )
ax1.set_title('Scatter: $x$ versus $y$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')

ax2.plot(k,MAEtopsis2AlphaFull, label='Topsis')
ax2.plot(k,MAEknn2AlphaFull_IGN_SIM,'r--',label='KNN Ignorance')
ax2.set_title('Scatter: $x$ versus $y$')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')
ax2.legend(loc=(0.65, 0.8))
ax2.plot()

ax3.hist(data, bins=np.arange(data.min(), data.max()),label=('x', 'y'))
ax3.legend(loc=(0.65, 0.8))
ax3.set_title('Frequencies of $x$ and $y$')
#ax3.yaxis.tick_right()




plt.show()
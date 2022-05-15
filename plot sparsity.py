import matplotlib.pyplot as plt
#plt.plot([1, 2, 3, 4])
import numpy as np

#****************************************************************K**************************************
s=[0.6097030752916225, 0.6533404029692471, 0.7010074231177095, 0.7506362672322375, 0.8082184517497348]

MAEknn_V1=[0.5023017942311522, 0.5383278176887737, 0.5131786289315826, 0.5446584961042005, 0.5199614860054061]
RMSEknn_V1=[0.659884510268091, 0.7071627958206945, 0.6829315265729797, 0.686632847717018, 0.660280532161734]
RCLknn_V1=[0.9422885572139303, 0.9309967141292442, 0.9484662576687116, 0.9481865284974094, 0.9570552147239264]
PRCknn_V1=[0.9088291746641075, 0.9249183895538629, 0.9290865384615384, 0.8926829268292683, 0.9052224371373307]
F2knn_V1=[0.925256472887152, 0.9279475982532751, 0.9386763812993321, 0.9195979899497487, 0.9304174950298211]

MAEknn_V2=[0.597860429477141, 0.6292691922927003, 0.59098768169006, 0.6096814645058954, 0.5601159394260358]
RMSEknn_V2=[0.7763811441463587, 0.8109734939689764, 0.7746079525916999, 0.7641578447589151, 0.7112662023950387]
RCLknn_V2=[0.9255213505461768, 0.9111592632719393, 0.9315403422982885, 0.9332191780821918, 0.9487704918032787]
PRCknn_V2=[0.9031007751937985, 0.9211391018619934, 0.9292682926829269, 0.8876221498371335, 0.904296875]
F2knn_V2=[0.9141736145169201, 0.9161220043572985, 0.9304029304029304, 0.9098497495826378, 0.9259999999999999]

MAEknn_V3=[0.47858362134216986, 0.5053051944285722, 0.4789132804966781, 0.51105404809106, 0.479921187355105]
RMSEknn_V3=[0.6314880238297833, 0.6680664201769958, 0.6443480942193232, 0.6469140163502856, 0.613621897880372]
RCLknn_V3=[0.9497991967871486, 0.9392265193370166, 0.9498164014687882, 0.9532871972318339, 0.9574898785425101]
PRCknn_V3=[0.904397705544933, 0.9169363538295577, 0.9349397590361446, 0.893030794165316, 0.900952380952381]
F2knn_V3=[0.9265426052889324, 0.9279475982532751, 0.9423193685488767, 0.9221757322175732, 0.9283611383709519]

MAEtopsis=[0.47674452610304463, 0.5070828796687066, 0.47691062134696866, 0.5080051144169303, 0.4687265905725497]
RMSEtopsis=[0.6313268211302526, 0.6736371327427784, 0.6430397705648797, 0.6444237082361262, 0.600254859569791]
RCLtopsis=[0.9507537688442211, 0.9351648351648352, 0.9532595325953259, 0.95, 0.9633401221995926]
PRCtopsis=[0.9061302681992337, 0.9239956568946797, 0.9303721488595438, 0.884430176565008, 0.8992395437262357]
F2topsis=[0.9279058361942129, 0.9295466957946477, 0.9416767922235723, 0.9160432252701578, 0.9301868239921336]

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

"""
fig, (ax1, ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=4,figsize=(20, 4))

#ax1.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')
ax1.plot(s, MAEtopsis, label='Topsis')
ax1.plot(s, MAEknn_V1, 'g-s', label='KNN V1')
ax1.plot(s, MAEknn_V2, 'r-o', label='KNN V2')
ax1.plot(s, MAEknn_V3, 'r--', label='KNN V3')
ax1.set_title('MAE')
ax1.set_xlabel('$k$')
ax1.set_ylabel('$MAE$')
ax1.legend(loc=(0.65, 0.8))

ax2.plot(s, RMSEtopsis, label='Topsis')
ax2.plot(s, RMSEknn_V1, 'g-s', label='KNN V1')
ax2.plot(s, RMSEknn_V2, 'r-o', label='KNN V2')
ax2.plot(s, RMSEknn_V3, 'r--', label='KNN V3')
ax2.set_title('RMSE')
ax2.set_xlabel('$k$')
ax2.set_ylabel('$RMSE$')
ax2.legend(loc=(0.65, 0.8))
ax2.plot()

ax3.plot(s, RCLtopsis, label='Topsis')
ax3.plot(s, RCLknn_V1, 'g-s', label='KNN V1')
ax3.plot(s, RCLknn_V2, 'r-o', label='KNN V2')
ax3.plot(s, RCLknn_V3, 'r--', label='KNN V3')
ax3.set_xlabel('$k$')
ax3.set_ylabel('$Recall$')
ax3.legend(loc=(0.65, 0.8))
ax3.set_title('Recall')
#ax3.yaxis.tick_right()


ax4.plot(s, PRCtopsis, label='Topsis')
ax4.plot(s, PRCknn_V1, 'g-s', label='KNN V1')
ax4.plot(s, PRCknn_V2, 'r-o', label='KNN V2')
ax4.plot(s, PRCknn_V3, 'r--', label='KNN V3')
ax4.set_xlabel('$k$')
ax4.set_ylabel('$PRC$')
ax4.legend(loc=(0.65, 0.8))
ax4.set_title('PRC')
plt.show()

"""
plt.plot(s, F2topsis,"k-d", label='Topsis')
plt.plot(s, F2knn_V1, 'm-s', label='KNN V1')
plt.plot(s, F2knn_V2, 'b-o', label='KNN V2')
plt.plot(s, F2knn_V3, 'y-*', label='KNN V3')
plt.suptitle('$F_{mesure}$')
#plt.set_title('MAE')
plt.xlabel('$s$')
plt.ylabel('$F_{mesure}$')
plt.xlim(0.55,0.85)
plt.legend()
plt.show()
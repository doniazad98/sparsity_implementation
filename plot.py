import matplotlib.pyplot as plt
#plt.plot([1, 2, 3, 4])
import numpy as np

#****************************************************************K**************************************
k=[10, 20, 30, 40,50]
"""
# old results
MAEtopsis=[0.7092180589206517,0.715208479218115,0.7200974076378949,0.7246577876264518,0.7281181809013414 ]
MAEknn_V1=[0.6982859509740496,0.7045871902004741,0.712131101322214,0.7177340363595236,0.7221699299632558]
MAEknn_V2=[0.7567831935508517,0.7465209157935389 ,0.7443842442902066,0.7426057308287877,0.7412579507605497]
MAEknn_V3=[0.6869115188702196,0.6977519888498929,0.7060548544851095,0.7128301006210693,0.7184374172672827]


RMSEtopsis=[0.9154018009752194,0.9196401601765144,0.9247779230144804,0.9290215141513427,0.9328973609135072]
RMSEknn_V1=[0.9016545937571814,0.9078382552579902,0.915718912208185,0.9213436392513484,0.9263872558093768]
RMSEknn_V2=[0.9707300801333782,0.9556680082397955 ,0.9509774152333262,0.9488137537897872 ,0.9469694189847493]
RMSEknn_V3=[0.8901898833311387,0.8995139732919517,0.9085506709679283,0.9157997366892542,0.9220670851221893]


RCLtopsis=[0.8739023533544081,0.866991643454039 ,0.8635970226761295,0.8626297577854671,0.8601965178417514 ]
RCLknn_V1=[0.8836709087650116 ,0.8768551236749117,0.8733555516576039,0.8690496948561465,0.8659722222222223 ]
RCLknn_V2=[0.8484069886947585,0.8512354152367879 , 0.8527997251803504, 0.8537298040563768,0.8556931544547643]
RCLknn_V3=[0.8875314408911247 ,0.8803041018387553,0.8762071992976295,0.8708551483420593,0.8670258995306797  ]


PRCtopsis=[0.7337068711294603,0.7439498057962355 ,0.7464093357271095 ,0.7492111194590534 ,0.7490243170219153 ]
PRCknn_V1=[0.7199182242990654 ,0.7298529411764706 ,0.7388336548449325,0.7436586093703372,0.7458133971291866 ]
PRCknn_V2=[0.7476226415094339,0.7500755972180224 ,0.748868778280543,0.7491704374057315,0.7510567632850241 ]
PRCknn_V3=[0.7160458037396724,0.7294169352475828,0.7370753323485968 , 0.7412358882947119,0.7440334128878282 ]
"""


# new  results :
MAEknn_V2=[0.7567831935508517, 0.7465209157935389, 0.7443842442902066, 0.7426057308287877, 0.7412579507605497]
RMSEknn_V2=[0.9707300801333782, 0.9556680082397955, 0.9509774152333262, 0.9488137537897872, 0.9469694189847493]
RCLknn_V2=[0.8484069886947585, 0.8512354152367879, 0.8527997251803504, 0.8537298040563768, 0.8556931544547643]
PRCknn_V2=[0.7476226415094339, 0.7500755972180224, 0.748868778280543, 0.7491704374057315, 0.7510567632850241]
F2knn_V2=[0.7948327048062264, 0.7974602153994533, 0.7974622550594281, 0.7980398457583547, 0.7999678404888246]


MAEknn_V3=[0.6869115188702196, 0.6977519888498929, 0.7060548544851095, 0.7128301006210693, 0.7184374172672827]
RMSEknn_V3=[0.8901898833311387, 0.8995139732919517, 0.9085506709679283, 0.9157997366892542, 0.9220670851221893]
RCLknn_V3=[0.8875314408911247, 0.8803041018387553, 0.8762071992976295, 0.8708551483420593, 0.8670258995306797]
PRCknn_V3=[0.7160458037396724, 0.7294169352475828, 0.7370753323485968, 0.7412358882947119, 0.7440334128878282]
F2knn_V3=[0.7926193341355796, 0.7977888158948887, 0.8006417970316887, 0.800834536992457, 0.8008348719595408]


MAEknn_V1=[0.6982859509740496, 0.7045871902004741, 0.712131101322214, 0.7177340363595236, 0.7221699299632558]
RMSEknn_V1=[0.9016545937571814, 0.9078382552579902, 0.915718912208185, 0.9213436392513484, 0.9263872558093768]
RCLknn_V1=[0.8836709087650116, 0.8768551236749117, 0.8733555516576039, 0.8690496948561465, 0.8659722222222223]
PRCknn_V1=[0.7199182242990654, 0.7298529411764706, 0.7388336548449325, 0.7436586093703372, 0.7458133971291866]
F2knn_V1=[0.7934336525307798, 0.7966292134831462, 0.8004823151125403, 0.8014794564605613, 0.8014138817480719]

MAEtopsis=[0.7092180589206517, 0.715208479218115, 0.7200974076378949, 0.7246577876264518, 0.7281181809013414]
RMSEtopsis=[0.9154018009752194, 0.9196401601765144, 0.9247779230144804, 0.9290215141513427, 0.9328973609135072]
RCLtopsis=[0.8739023533544081, 0.866991643454039, 0.8635970226761295, 0.8626297577854671, 0.8601965178417514]
PRCtopsis=[0.7337068711294603, 0.7439498057962355, 0.7464093357271095, 0.7492111194590534, 0.7490243170219153]
F2topsis=[0.7976915678101956, 0.8007718282682104, 0.8007383035069416, 0.8019300361881785, 0.8007702800288855]

MAEknn_V30=[0.4105986559593819, 0.43348107527324786, 0.45564929815322625, 0.4693298236057459, 0.4789132804966781]
RMSEknn_V30=[0.5612561960107196, 0.5881975310798692, 0.6150460689520149, 0.6319803929075292, 0.6443480942193232]
RCLknn_V30=[0.97, 0.9651741293532339, 0.9568434032059187, 0.9568434032059187, 0.9498164014687882]
PRCknn_V30=[0.9044289044289044, 0.9140164899882215, 0.9216152019002375, 0.9338146811070999, 0.9349397590361446]
F2knn_V30=[0.9360675512665863, 0.9388989715668482, 0.9388989715668482, 0.9451887941534715, 0.9423193685488767]

MAEknn_V20=[0.5663277012507377, 0.5791946551480189, 0.583424861034145, 0.586284855813006, 0.59098768169006]
RMSEknn_V20=[0.7488317284408441, 0.7608983432244085, 0.7669248094867748, 0.7693591351482444, 0.7746079525916999]
RCLknn_V20=[0.9420468557336621, 0.9405940594059405, 0.93711467324291, 0.9339045287637698, 0.9315403422982885]
PRCknn_V20=[0.9062870699881376, 0.9134615384615384, 0.9123649459783914, 0.9214975845410628, 0.9292682926829269]
F2knn_V20=[0.9238210399032648, 0.926829268292683, 0.924574209245742, 0.9276595744680851, 0.9304029304029304]

MAEknn_V10=[0.4438032888310604, 0.4735498956290337, 0.49065528842453315, 0.5017707252665645, 0.5131786289315826]
RMSEknn_V10=[0.6028077837671423, 0.6371629830717134, 0.6587098952709743, 0.6708283376705143, 0.6829315265729797]
RCLknn_V10=[0.9639303482587065, 0.9580246913580247, 0.9567367119901112, 0.9485294117647058, 0.9484662576687116]
PRCknn_V10=[0.9064327485380117, 0.9227110582639715, 0.9258373205741627, 0.9269461077844311, 0.9290865384615384]
F2knn_V10=[0.9342977697408078, 0.9400363416111448, 0.941033434650456, 0.9376135675348274, 0.9386763812993321]

MAEtopsis0=[0.4104704841265994, 0.4292648729977242, 0.45039455196384337, 0.46666049396768317, 0.47691062134696866]
RMSEtopsis0=[0.5667899652540208, 0.589041772095948, 0.612464159270021, 0.6307703999513291, 0.6430397705648797]
RCLtopsis0=[0.9676214196762142, 0.9663760896637609, 0.9580246913580247, 0.9556650246305419, 0.9532595325953259]
PRCtopsis0=[0.8993055555555556, 0.9161747343565525, 0.9238095238095239, 0.9293413173652695, 0.9303721488595438]
F2topsis0=[0.9322135572885423, 0.9406060606060606, 0.9406060606060606, 0.9423193685488768, 0.9416767922235723]


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
x = np.random.randint(low=1, high=11, size=50)
y = x + np.random.randint(1, 5, size=x.size)
data = np.column_stack((x, y))

fig, (ax1, ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=4,figsize=(20, 4))

#ax1.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')
ax1.plot(k,MAEtopsis,label='Topsis' )
ax1.plot(k,MAEknn_V1,'g-s',label='KNN V1' )
ax1.plot(k,MAEknn_V2,'r-o',label='KNN V2' )
ax1.plot(k,MAEknn_V3,'r--',label='KNN V3')
ax1.set_title('MAE')
ax1.set_xlabel('$k$')
ax1.set_ylabel('$MAE$')
ax1.legend(loc=(0.65, 0.8))

ax2.plot(k,RMSEtopsis,label='Topsis' )
ax2.plot(k,RMSEknn_V1,'g-s',label='KNN V1' )
ax2.plot(k,RMSEknn_V2,'r-o',label='KNN V2' )
ax2.plot(k,RMSEknn_V3,'r--',label='KNN V3')
ax2.set_title('RMSE')
ax2.set_xlabel('$k$')
ax2.set_ylabel('$RMSE$')
ax2.legend(loc=(0.65, 0.8))
ax2.plot()

ax3.plot(k,RCLtopsis,label='Topsis' )
ax3.plot(k,RCLknn_V1,'g-s',label='KNN V1' )
ax3.plot(k,RCLknn_V2,'r-o',label='KNN V2' )
ax3.plot(k,RCLknn_V3,'r--',label='KNN V3')
ax3.set_xlabel('$k$')
ax3.set_ylabel('$Recall$')
ax3.legend(loc=(0.65, 0.8))
ax3.set_title('Recall')
#ax3.yaxis.tick_right()


ax4.plot(k,PRCtopsis,label='Topsis' )
ax4.plot(k,PRCknn_V1,'g-s',label='KNN V1' )
ax4.plot(k,PRCknn_V2,'r-o',label='KNN V2' )
ax4.plot(k,PRCknn_V3,'r--',label='KNN V3')
ax4.set_xlabel('$k$')
ax4.set_ylabel('$PRC$')
ax4.legend(loc=(0.65, 0.8))
ax4.set_title('PRC')


plt.show()


plt.plot(k, F2topsis,"k-d", label='Topsis')
plt.plot(k, F2knn_V1, 'm-s', label='KNN V1')
plt.plot(k, F2knn_V2, 'b-o', label='KNN V2')
plt.plot(k, F2knn_V3, 'y-*', label='KNN V3')
plt.suptitle('$F_{mesure}$')
#plt.set_title('MAE')
plt.xlabel('$k$')
plt.ylabel('$F_{mesure}$')
#plt.xlim(0.55,0.85)
plt.legend()
"""

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
plot1=plt.figure(1)
ax1.plot(k, MAEtopsis,"k-d", label='Topsis')
ax1.plot(k, MAEknn_V1, 'm-s', label='KNN V1')
ax1.plot(k, MAEknn_V2, 'b-o', label='KNN V2')
ax1.plot(k, MAEknn_V3, 'y-*', label='KNN V3')
#ax1.suptitle('Parcimonie  60,97')
ax1.set_title('Parcimonie  93.49')
ax1.set_xlabel('$k$')
ax1.set_ylabel('$MAE$')
#ax1.xlim(0.55,0.85)
ax1.legend()
ax2.plot(k, MAEtopsis0,"k-d", label='Topsis')
ax2.plot(k, MAEknn_V10, 'm-s', label='KNN V1')
ax2.plot(k, MAEknn_V20, 'b-o', label='KNN V2')
ax2.plot(k, MAEknn_V30, 'y-*', label='KNN V3')

ax2.set_title('Parcimonie  70,10')
ax2.set_xlabel('$k$')
ax2.set_ylabel('$MAE$')
#ax2.xlim(0.55,0.85)
ax2.legend()


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
plot1=plt.figure(1)
ax1.plot(k, RMSEtopsis,"k-d", label='Topsis')
ax1.plot(k, RMSEknn_V1, 'm-s', label='KNN V1')
ax1.plot(k, RMSEknn_V2, 'b-o', label='KNN V2')
ax1.plot(k, RMSEknn_V3, 'y-*', label='KNN V3')
#ax1.suptitle('Parcimonie  60,97')
ax1.set_title('Parcimonie  93.49')
ax1.set_xlabel('$k$')
ax1.set_ylabel('$RMSE$')
#ax1.xlim(0.55,0.85)
ax1.legend()
ax2.plot(k, RMSEtopsis0,"k-d", label='Topsis')
ax2.plot(k, RMSEknn_V10, 'm-s', label='KNN V1')
ax2.plot(k, RMSEknn_V20, 'b-o', label='KNN V2')
ax2.plot(k, RMSEknn_V30, 'y-*', label='KNN V3')

ax2.set_title('Parcimonie  70,10')
ax2.set_xlabel('$k$')
ax2.set_ylabel('$RMSE$')
#ax2.xlim(0.55,0.85)
ax2.legend()


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
plot1=plt.figure(1)
ax1.plot(k, RCLtopsis,"k-d", label='Topsis')
ax1.plot(k, RCLknn_V1, 'm-s', label='KNN V1')
ax1.plot(k, RCLknn_V2, 'b-o', label='KNN V2')
ax1.plot(k, RCLknn_V3, 'y-*', label='KNN V3')
#ax1.suptitle('Parcimonie  60,97')
ax1.set_title('Parcimonie  93.49')
ax1.set_xlabel('$k$')
ax1.set_ylabel('$Recall$')
#ax1.xlim(0.55,0.85)
ax1.legend()
ax2.plot(k, RCLtopsis0,"k-d", label='Topsis')
ax2.plot(k, RCLknn_V10, 'm-s', label='KNN V1')
ax2.plot(k, RCLknn_V20, 'b-o', label='KNN V2')
ax2.plot(k, RCLknn_V30, 'y-*', label='KNN V3')

ax2.set_title('Parcimonie  70,10')
ax2.set_xlabel('$k$')
ax2.set_ylabel('$Recall$')
#ax2.xlim(0.55,0.85)
ax2.legend()


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
plot1=plt.figure(1)
ax1.plot(k, PRCtopsis,"k-d", label='Topsis')
ax1.plot(k, PRCknn_V1, 'm-s', label='KNN V1')
ax1.plot(k, PRCknn_V2, 'b-o', label='KNN V2')
ax1.plot(k, PRCknn_V3, 'y-*', label='KNN V3')
#ax1.suptitle('Parcimonie  60,97')
ax1.set_title('Parcimonie  93.49')
ax1.set_xlabel('$k$')
ax1.set_ylabel('$Precision$')
#ax1.xlim(0.55,0.85)
ax1.legend()
ax2.plot(k, PRCtopsis0,"k-d", label='Topsis')
ax2.plot(k, PRCknn_V10, 'm-s', label='KNN V1')
ax2.plot(k, PRCknn_V20, 'b-o', label='KNN V2')
ax2.plot(k, PRCknn_V30, 'y-*', label='KNN V3')

ax2.set_title('Parcimonie  70,10')
ax2.set_xlabel('$k$')
ax2.set_ylabel('$Precision$')
#ax2.xlim(0.55,0.85)
ax2.legend()


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
plot1=plt.figure(1)
ax1.plot(k, F2topsis,"k-d", label='Topsis')
ax1.plot(k, F2knn_V1, 'm-s', label='KNN V1')
ax1.plot(k, F2knn_V2, 'b-o', label='KNN V2')
ax1.plot(k, F2knn_V3, 'y-*', label='KNN V3')
#ax1.suptitle('Parcimonie  60,97')
ax1.set_title('Parcimonie  93.49')
ax1.set_xlabel('$k$')
ax1.set_ylabel('$F measure$')
#ax1.xlim(0.55,0.85)
ax1.legend()
ax2.plot(k, F2topsis0,"k-d", label='Topsis')
ax2.plot(k, F2knn_V10, 'm-s', label='KNN V1')
ax2.plot(k, F2knn_V20, 'b-o', label='KNN V2')
ax2.plot(k, F2knn_V30, 'y-*', label='KNN V3')

ax2.set_title('Parcimonie  70,10')
ax2.set_xlabel('$k$')
ax2.set_ylabel('$F measure$')
#ax2.xlim(0.55,0.85)
ax2.legend()

plt.show()
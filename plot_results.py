import sys
import traceback
import numpy as np
from scipy.stats import binom
# sys.path.append('..')
from utils.utils import *
from mne import create_info, EvokedArray
from mne.viz import plot_topomap
from mne.channels import read_custom_montage

output_type = 'pdf'
output_path = os.path.abspath('./results')
save_path_EventAAD_SI_acc_box = os.path.join(output_path, f"fig_EventAAD_SI_acc_box.{output_type}")
save_path_EventAAD_SI_mesd = os.path.join(output_path, f"fig_EventAAD_SI_mesd.{output_type}")
save_path_Fuglsang_SI_acc_box = os.path.join(output_path, f"fig_DTU_SI_acc_box.{output_type}")
save_path_Fuglsang_SI_mesd = os.path.join(output_path, f"fig_DTU_SI_mesd.{output_type}")
save_path_KULeuven_SI_acc_box = os.path.join(output_path, f"fig_KULeuven_SI_acc_box.{output_type}")
save_path_KULeuven_SI_mesd = os.path.join(output_path, f"fig_KULeuven_SI_mesd.{output_type}")

save_path_EventAAD_SS_acc_box = os.path.join(output_path, f"fig_EventAAD_SS_acc_box.{output_type}")
save_path_EventAAD_SS_mesd = os.path.join(output_path, f"fig_EventAAD_SS_mesd.{output_type}")
save_path_Fuglsang_SS_acc_box = os.path.join(output_path, f"fig_DTU_SS_acc_box.{output_type}")
save_path_Fuglsang_SS_mesd = os.path.join(output_path, f"fig_DTU_SS_mesd.{output_type}")
save_path_KULeuven_SS_acc_box = os.path.join(output_path, f"fig_KULeuven_SS_acc_box.{output_type}")
save_path_KULeuven_SS_mesd = os.path.join(output_path, f"fig_KULeuven_SS_mesd.{output_type}")

x_label = 'Window length (s)'
y_label = 'Accuracy'
WINDOWS = np.array([1, 2, 5, 10, 20, 40])
WINDOWS_EventAAD = np.array([1, 2, 5, 10, 20])
WINDOWS_Fuglsang = np.array([1, 2, 5, 10, 20, 40])
WINDOWS_KULeuven = np.array([1, 2, 5, 10, 20, 40])

model_names = ['LSR', 'CCA', 'NSR', 'AADNet']

title_acc_EventAAD = f'Comparison of accuracy on EventAAD dataset.'
title_mesd_EventAAD = f'Comparison of MESD on EventAAD dataset.'
title_acc_Fuglsang = f'Comparison of accuracy on DTU dataset.'
title_mesd_Fuglsang = f'Comparison of MESD on DTU dataset.'
title_acc_KULeuven = f'Comparison of accuracy on KUL dataset.'
title_mesd_KULeuven = f'Comparison of MESD on KUL dataset.'

test_size_EventAAD = np.array([826, 428, 180, 102, 62])
test_size_Fuglsang = np.array([2460, 1260, 540, 300, 180, 120])
test_size_KULeuven = np.array([4444, 2228, 890, 452, 228, 116])
# chance levels
cl_eventaad = binom.ppf(0.95, test_size_EventAAD, 0.5)/test_size_EventAAD
cl_fuglsang = binom.ppf(0.95, test_size_Fuglsang, 0.5)/test_size_Fuglsang
cl_KULeuven = binom.ppf(0.95, test_size_KULeuven, 0.5)/test_size_KULeuven


SI_files_eventaad = [
                     'LOSO_LSQ_EventAAD_final_SI_acc.npy', # final
                     'LOSO_CCA_EventAAD_final_SI_acc.npy', # final
                     'LOSO_NSR_EventAAD_final_SI_acc.npy',
                     'LOSO_AADNet_EventAAD_final_SI_acc.npy',
                     ]

SI_files_dtu = [
                'LOSO_LSQ_DTU_final_SI_acc.npy', # final
                'LOSO_CCA_DTU_final_SI_acc.npy', # final
                'LOSO_NSR_DTU_final_SI_acc.npy',
                'LOSO_AADNet_DTU_final_SI_acc.npy',
                ]

SI_files_KULeuven = [ # exact cross_validation
                'LOSO_LSQ_Das_final_SI_acc.npy', #final
                'LOSO_CCA_Das_final_SI_acc.npy', # final
                'LOSO_NSR_Das_final_SI_acc.npy',
                'LOSO_AADNet_Das_final_SI_acc.npy',
                ]
                
SS_files_eventaad = [
                     'SS_LSQ_EventAAD_final_SS_acc.npy', # final
                     'SS_CCA_EventAAD_final_SS_acc.npy', #final
                     'SS_NSR_EventAAD_final_SS_acc.npy',
                     'SS_AADNet_EventAAD_final_SS_acc.npy',
                     ]

SS_files_dtu = [
                'SS_LSQ_DTU_final_SS_acc.npy', # final
                'SS_CCA_DTU_final_SS_acc.npy', # final
                'SS_NSR_DTU_final_SS_acc.npy',
                'SS_AADNet_DTU_final_SS_acc.npy',
                ]
                
SS_files_KULeuven = [# exact cross_validation
                'SS_LSQ_Das_final_SS_acc.npy', #final
                'SS_CCA_Das_final_SS_acc.npy', # final
                'SS_NSR_Das_final_SS_acc.npy',
                'SS_AADNet_Das_final_SS_acc.npy',
                ]                

EventAAD_SI = []
Fuglsang_SI = []
KULeuven_SI = []
EventAAD_SS = []
Fuglsang_SS = []
KULeuven_SS = []
for i in range(len(SI_files_eventaad)):
    EventAAD_SI.append(np.load(os.path.join(output_path, SI_files_eventaad[i]))[0])
for i in range(len(SI_files_dtu)):
    Fuglsang_SI.append(np.load(os.path.join(output_path, SI_files_dtu[i]))[0])
for i in range(len(SI_files_KULeuven)):
    KULeuven_SI.append(np.load(os.path.join(output_path, SI_files_KULeuven[i]))[0])
for i in range(len(SS_files_eventaad)):
    EventAAD_SS.append(np.load(os.path.join(output_path, SS_files_eventaad[i]))[0])    
for i in range(len(SS_files_dtu)):
    Fuglsang_SS.append(np.load(os.path.join(output_path, SS_files_dtu[i]))[0])
for i in range(len(SS_files_KULeuven)):
    KULeuven_SS.append(np.load(os.path.join(output_path, SS_files_KULeuven[i]))[0])
    
# calculate MESD
mesd_EventAAD_SI = np.zeros((len(EventAAD_SI), 25))
mesd_Fuglsang_SI = np.zeros((len(Fuglsang_SI), 19))
mesd_KULeuven_SI = np.zeros((len(KULeuven_SI), 17))
mesd_EventAAD_SS = np.zeros((len(EventAAD_SS), 25))
mesd_Fuglsang_SS = np.zeros((len(Fuglsang_SS), 19))
mesd_KULeuven_SS = np.zeros((len(KULeuven_SS), 17))

# count elements < 0.5
for i in range(len(model_names)):    
    EventAAD_SI[i][EventAAD_SI[i] < 0.5] = 0.51
    Fuglsang_SI[i][Fuglsang_SI[i] < 0.5] = 0.51
    KULeuven_SI[i][KULeuven_SI[i] < 0.5] = 0.51
    
    EventAAD_SI[i][EventAAD_SI[i] == 1.0] = 0.99
    Fuglsang_SI[i][Fuglsang_SI[i] == 1.0] = 0.99
    KULeuven_SI[i][KULeuven_SI[i] == 1.0] = 0.99
    
    EventAAD_SS[i][EventAAD_SS[i] < 0.5] = 0.51
    Fuglsang_SS[i][Fuglsang_SS[i] < 0.5] = 0.51
    KULeuven_SS[i][KULeuven_SS[i] < 0.5] = 0.51
    
    EventAAD_SS[i][EventAAD_SS[i] == 1.0] = 0.99
    Fuglsang_SS[i][Fuglsang_SS[i] == 1.0] = 0.99
    KULeuven_SS[i][KULeuven_SS[i] == 1.0] = 0.99    

for i in range(len(model_names)):
    for j in range(EventAAD_SI[i].shape[1]):
        try:
            mesd_EventAAD_SI[i,j], *_ = compute_MESD(WINDOWS_EventAAD, EventAAD_SI[i][:,j], N_min=5, P0=0.8, c=0.65)
        except AssertionError:
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]            
            print(f'got AssertionError: {text}')
            continue
    for j in range(Fuglsang_SI[i].shape[1]):
        try:
            mesd_Fuglsang_SI[i,j], *_ = compute_MESD(WINDOWS_Fuglsang, Fuglsang_SI[i][:,j], N_min=5, P0=0.8, c=0.65)
        except AssertionError:
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]            
            print(f'got AssertionError: {text}')
            continue
            
    for j in range(KULeuven_SI[i].shape[1]):
        try:
            mesd_KULeuven_SI[i,j], *_ = compute_MESD(WINDOWS_KULeuven, KULeuven_SI[i][:,j], N_min=5, P0=0.8, c=0.65)
        except AssertionError:
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]            
            print(f'got AssertionError: {text}')
            continue            

    try:
        mesd_EventAAD_SI[i,-1], *_ = compute_MESD(WINDOWS_EventAAD, EventAAD_SI[i].mean(axis=1, keepdims=False), N_min=5, P0=0.8, c=0.65)
        mesd_Fuglsang_SI[i,-1], *_ = compute_MESD(WINDOWS_Fuglsang, Fuglsang_SI[i].mean(axis=1, keepdims=False), N_min=5, P0=0.8, c=0.65)
        mesd_KULeuven_SI[i,-1], *_ = compute_MESD(WINDOWS_KULeuven, KULeuven_SI[i].mean(axis=1, keepdims=False), N_min=5, P0=0.8, c=0.65)
    except AssertionError:
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]            
        print(f'got AssertionError: {text}')
        # continue       

print(f'median of mesd EventAAD_SI: {np.median(mesd_EventAAD_SI[:,:-1], axis=1, keepdims=False)}')
print(f'median of mesd Fuglsang_SI: {np.median(mesd_Fuglsang_SI[:,:-1], axis=1, keepdims=False)}')
print(f'median of mesd KULeuven_SI: {np.median(mesd_KULeuven_SI[:,:-1], axis=1, keepdims=False)}')
                
# statistical test
stats_EventAAD_SI = np.zeros((len(EventAAD_SI)-1, 6))
stats_Fuglsang_SI = np.zeros((len(Fuglsang_SI)-1, 6))
stats_KULeuven_SI = np.zeros((len(KULeuven_SI)-1, 6))
mesd_stats_EventAAD_SI = np.zeros(len(EventAAD_SI)-1)
mesd_stats_Fuglsang_SI = np.zeros(len(Fuglsang_SI)-1)
mesd_stats_KULeuven_SI = np.zeros(len(KULeuven_SI)-1)

for i in range(len(EventAAD_SI)-1):
    for j in range(len(EventAAD_SI[i])):    
        stats_EventAAD_SI[i,j] = permutation_test(EventAAD_SI[-1][j]-EventAAD_SI[i][j], EventAAD_SI[i][j]-EventAAD_SI[i][j], tail=1)
    mesd_stats_EventAAD_SI[i] = permutation_test(mesd_EventAAD_SI[-1][:-1]-mesd_EventAAD_SI[i][:-1], mesd_EventAAD_SI[i][:-1]-mesd_EventAAD_SI[i][:-1], tail=-1, statistics='median')
    
for i in range(len(Fuglsang_SI)-1):
    for j in range(len(Fuglsang_SI[i])):    
        stats_Fuglsang_SI[i,j] = permutation_test(Fuglsang_SI[-1][j]-Fuglsang_SI[i][j], Fuglsang_SI[i][j]-Fuglsang_SI[i][j], tail=1)
    mesd_stats_Fuglsang_SI[i] = permutation_test(mesd_Fuglsang_SI[-1][:-1]-mesd_Fuglsang_SI[i][:-1], mesd_Fuglsang_SI[i][:-1]-mesd_Fuglsang_SI[i][:-1], tail=-1, statistics='median')
    
for i in range(len(KULeuven_SI)-1):
    for j in range(len(KULeuven_SI[i])):    
        stats_KULeuven_SI[i,j] = permutation_test(KULeuven_SI[-1][j]-KULeuven_SI[i][j], KULeuven_SI[i][j]-KULeuven_SI[i][j], tail=1)
    mesd_stats_KULeuven_SI[i] = permutation_test(mesd_KULeuven_SI[-1][:-1]-mesd_KULeuven_SI[i][:-1], mesd_KULeuven_SI[i][:-1]-mesd_KULeuven_SI[i][:-1], tail=-1, statistics='median')    

# plot
plot_box_barSTD(EventAAD_SI, box_labels=model_names, xtick_labels=WINDOWS_EventAAD, x_label=x_label, y_label=y_label, title=title_acc_EventAAD, stats=stats_EventAAD_SI*3, chance_lv=cl_eventaad, save_path=save_path_EventAAD_SI_acc_box)
plot_mesd(acc=EventAAD_SI, line_labels=model_names, xtick_labels=WINDOWS_EventAAD, title=title_mesd_EventAAD, save_path=save_path_EventAAD_SI_mesd, stats=mesd_stats_EventAAD_SI*3)

plot_box_barSTD(Fuglsang_SI, box_labels=model_names, xtick_labels=WINDOWS_Fuglsang, x_label=x_label, y_label=y_label, title=title_acc_Fuglsang, stats=stats_Fuglsang_SI*3, chance_lv=cl_fuglsang, save_path=save_path_Fuglsang_SI_acc_box)
plot_mesd(acc=Fuglsang_SI, line_labels=model_names, xtick_labels=WINDOWS_Fuglsang, title=title_mesd_Fuglsang, save_path=save_path_Fuglsang_SI_mesd, stats=mesd_stats_Fuglsang_SI*3)

plot_box_barSTD(KULeuven_SI, box_labels=model_names, xtick_labels=WINDOWS_KULeuven, x_label=x_label, y_label=y_label, title=title_acc_KULeuven, stats=stats_KULeuven_SI*3, chance_lv=cl_fuglsang, save_path=save_path_KULeuven_SI_acc_box)
plot_mesd(acc=KULeuven_SI, line_labels=model_names, xtick_labels=WINDOWS_KULeuven, title=title_mesd_KULeuven, save_path=save_path_KULeuven_SI_mesd, stats=mesd_stats_KULeuven_SI*3)
  
# print
print(f'EventAAD_SI')
for i in range(len(EventAAD_SI)):
    print(f'{model_names[i]}:  {EventAAD_SI[i].mean(axis=1, keepdims=False).round(decimals=3)}')
print(f'stats_EventAAD_SI: {stats_EventAAD_SI*3}')
print(f'mesd_stats_EventAAD_SI: {mesd_stats_EventAAD_SI*3}')

#
print(f'Fuglsang_SI')
for i in range(len(Fuglsang_SI)):
    print(f'{model_names[i]}:  {Fuglsang_SI[i].mean(axis=1, keepdims=False).round(decimals=3)}')
print(f'stats_Fuglsang_SI: {stats_Fuglsang_SI*3}')
print(f'mesd_stats_Fuglsang_SI: {mesd_stats_Fuglsang_SI*3}')
#
print(f'KULeuven_SI')
for i in range(len(KULeuven_SI)):
    print(f'{model_names[i]}:  {KULeuven_SI[i].mean(axis=1, keepdims=False).round(decimals=3)}')
print(f'stats_KULeuven_SI: {stats_KULeuven_SI*3}')
print(f'mesd_stats_KULeuven_SI: {mesd_stats_KULeuven_SI*3}')

print('gap EventAAD SI')
for i in range(len(EventAAD_SI)-1):
    print(f'{model_names[i]}:  {EventAAD_SI[-1].mean(axis=1, keepdims=False).round(decimals=3) - EventAAD_SI[i].mean(axis=1, keepdims=False).round(decimals=3)}')
print('gap Fuglsang SI')
for i in range(len(Fuglsang_SI)-1):
    print(f'{model_names[i]}:  {Fuglsang_SI[-1].mean(axis=1, keepdims=False).round(decimals=3) - Fuglsang_SI[i].mean(axis=1, keepdims=False).round(decimals=3)}')      
print('gap KULeuven SI')
for i in range(len(KULeuven_SI)-1):
    print(f'{model_names[i]}:  {KULeuven_SI[-1].mean(axis=1, keepdims=False).round(decimals=3) - KULeuven_SI[i].mean(axis=1, keepdims=False).round(decimals=3)}')       

# For SS
for i in range(len(model_names)):
    for j in range(EventAAD_SS[i].shape[1]):
        try:
            mesd_EventAAD_SS[i,j], *_ = compute_MESD(WINDOWS_EventAAD, EventAAD_SS[i][:,j], N_min=5, P0=0.8, c=0.65)
        except AssertionError:
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]            
            print(f'got AssertionError: {text}')
            continue
    for j in range(Fuglsang_SS[i].shape[1]):
        try:
            mesd_Fuglsang_SS[i,j], *_ = compute_MESD(WINDOWS_Fuglsang, Fuglsang_SS[i][:,j], N_min=5, P0=0.8, c=0.65)
        except AssertionError:
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]            
            print(f'got AssertionError: {text}')
            continue
            
    for j in range(KULeuven_SS[i].shape[1]):
        try:
            mesd_KULeuven_SS[i,j], *_ = compute_MESD(WINDOWS_KULeuven, KULeuven_SS[i][:,j], N_min=5, P0=0.8, c=0.65)
        except AssertionError:
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]            
            print(f'got AssertionError: {text}')
            continue            

    try:
        mesd_EventAAD_SS[i,-1], *_ = compute_MESD(WINDOWS_EventAAD, EventAAD_SS[i].mean(axis=1, keepdims=False), N_min=5, P0=0.8, c=0.65)
        mesd_Fuglsang_SS[i,-1], *_ = compute_MESD(WINDOWS_Fuglsang, Fuglsang_SS[i].mean(axis=1, keepdims=False), N_min=5, P0=0.8, c=0.65)
        mesd_KULeuven_SS[i,-1], *_ = compute_MESD(WINDOWS_KULeuven, KULeuven_SS[i].mean(axis=1, keepdims=False), N_min=5, P0=0.8, c=0.65)
    except AssertionError:
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]            
        print(f'got AssertionError: {text}')
        # continue       
        
# statistical test
stats_EventAAD_SS = np.zeros((len(EventAAD_SS)-1, 6))
stats_Fuglsang_SS = np.zeros((len(Fuglsang_SS)-1, 6))
stats_KULeuven_SS = np.zeros((len(KULeuven_SS)-1, 6))
mesd_stats_EventAAD_SS = np.zeros(len(EventAAD_SS)-1)
mesd_stats_Fuglsang_SS = np.zeros(len(Fuglsang_SS)-1)
mesd_stats_KULeuven_SS = np.zeros(len(KULeuven_SS)-1)

for i in range(len(EventAAD_SS)-1):
    for j in range(len(EventAAD_SS[i])):    
        stats_EventAAD_SS[i,j] = permutation_test(EventAAD_SS[-1][j]-EventAAD_SS[i][j], EventAAD_SS[i][j]-EventAAD_SS[i][j], tail=1)
    mesd_stats_EventAAD_SS[i] = permutation_test(mesd_EventAAD_SS[-1][:-1]-mesd_EventAAD_SS[i][:-1], mesd_EventAAD_SS[i][:-1]-mesd_EventAAD_SS[i][:-1], tail=-1, statistics='median')
    
for i in range(len(Fuglsang_SS)-1):
    for j in range(len(Fuglsang_SS[i])):    
        stats_Fuglsang_SS[i,j] = permutation_test(Fuglsang_SS[-1][j]-Fuglsang_SS[i][j], Fuglsang_SS[i][j]-Fuglsang_SS[i][j], tail=1)
    mesd_stats_Fuglsang_SS[i] = permutation_test(mesd_Fuglsang_SS[-1][:-1]-mesd_Fuglsang_SS[i][:-1], mesd_Fuglsang_SS[i][:-1]-mesd_Fuglsang_SS[i][:-1], tail=-1, statistics='median')
    
for i in range(len(KULeuven_SS)-1):
    for j in range(len(KULeuven_SS[i])):    
        stats_KULeuven_SS[i,j] = permutation_test(KULeuven_SS[-1][j]-KULeuven_SS[i][j], KULeuven_SS[i][j]-KULeuven_SS[i][j], tail=1)
    mesd_stats_KULeuven_SS[i] = permutation_test(mesd_KULeuven_SS[-1][:-1]-mesd_KULeuven_SS[i][:-1], mesd_KULeuven_SS[i][:-1]-mesd_KULeuven_SS[i][:-1], tail=-1, statistics='median')    
    
# print
print(f'EventAAD_SS')
for i in range(len(EventAAD_SS)):
    print(f'{model_names[i]}:  {EventAAD_SS[i].mean(axis=1, keepdims=False).round(decimals=3)}')
print(f'stats_EventAAD_SS: {stats_EventAAD_SS*3}')
print(f'mesd_stats_EventAAD_SS: {mesd_stats_EventAAD_SS*3}')

#
print(f'Fuglsang_SS')
for i in range(len(Fuglsang_SS)):
    print(f'{model_names[i]}:  {Fuglsang_SS[i].mean(axis=1, keepdims=False).round(decimals=3)}')
print(f'stats_Fuglsang_SS: {stats_Fuglsang_SS*3}')
print(f'mesd_stats_Fuglsang_SS: {mesd_stats_Fuglsang_SS*3}')
#
print(f'KULeuven_SS')
for i in range(len(KULeuven_SS)):
    print(f'{model_names[i]}:  {KULeuven_SS[i].mean(axis=1, keepdims=False).round(decimals=3)}')
print(f'stats_KULeuven_SS: {stats_KULeuven_SS*3}')
print(f'mesd_stats_KULeuven_SS: {mesd_stats_KULeuven_SS*3}')

print(f'median of mesd EventAAD_SS: {np.median(mesd_EventAAD_SS[:,:-1], axis=1, keepdims=False)}')
print(f'median of mesd Fuglsang_SS: {np.median(mesd_Fuglsang_SS[:,:-1], axis=1, keepdims=False)}')
print(f'median of mesd KULeuven_SS: {np.median(mesd_KULeuven_SS[:,:-1], axis=1, keepdims=False)}')    

print('drop EventAAD')
for i in range(len(EventAAD_SI)):
    print(f'{model_names[i]}:  {(EventAAD_SS[i].mean(axis=1, keepdims=False).round(decimals=3) - EventAAD_SI[i].mean(axis=1, keepdims=False).round(decimals=3))*100}')
    
print('drop Fuglsang')
for i in range(len(Fuglsang_SI)):
    print(f'{model_names[i]}:  {(Fuglsang_SS[i].mean(axis=1, keepdims=False).round(decimals=3) - Fuglsang_SI[i].mean(axis=1, keepdims=False).round(decimals=3))*100}')
    
print('drop KULeuven')
for i in range(len(KULeuven_SI)):
    print(f'{model_names[i]}:  {(KULeuven_SS[i].mean(axis=1, keepdims=False).round(decimals=3) - KULeuven_SI[i].mean(axis=1, keepdims=False).round(decimals=3))*100}')    

# plot
plot_box_barSTD(EventAAD_SS, box_labels=model_names, xtick_labels=WINDOWS_EventAAD, x_label=x_label, y_label=y_label, title=title_acc_EventAAD, stats=stats_EventAAD_SS*3, chance_lv=cl_eventaad, save_path=save_path_EventAAD_SS_acc_box)
plot_mesd(acc=EventAAD_SS, line_labels=model_names, xtick_labels=WINDOWS_EventAAD, title=title_mesd_EventAAD, save_path=save_path_EventAAD_SS_mesd, stats=mesd_stats_EventAAD_SS*3)

plot_box_barSTD(Fuglsang_SS, box_labels=model_names, xtick_labels=WINDOWS_Fuglsang, x_label=x_label, y_label=y_label, title=title_acc_Fuglsang, stats=stats_Fuglsang_SS*3, chance_lv=cl_fuglsang, save_path=save_path_Fuglsang_SS_acc_box)
plot_mesd(acc=Fuglsang_SS, line_labels=model_names, xtick_labels=WINDOWS_Fuglsang, title=title_mesd_Fuglsang, save_path=save_path_Fuglsang_SS_mesd, stats=mesd_stats_Fuglsang_SS*3)

plot_box_barSTD(KULeuven_SS, box_labels=model_names, xtick_labels=WINDOWS_KULeuven, x_label=x_label, y_label=y_label, title=title_acc_KULeuven, stats=stats_KULeuven_SS*3, chance_lv=cl_fuglsang, save_path=save_path_KULeuven_SS_acc_box)
plot_mesd(acc=KULeuven_SS, line_labels=model_names, xtick_labels=WINDOWS_KULeuven, title=title_mesd_KULeuven, save_path=save_path_KULeuven_SS_mesd, stats=mesd_stats_KULeuven_SS*3)


# plot channel distribution
cmap = plt.get_cmap('jet')      

ori_performance_files = ['LOSO_AADNet_EventAAD_original.npy',
                         'LOSO_AADNet_DTU_original.npy',
                         'LOSO_AADNet_Das_original.npy',
                         ]   

distribution_files = ['channel_distribute_AADNet_EventAAD_LOCO_acc.npy',
                      'channel_distribute_AADNet_DTU_LOCO_acc.npy',
                      'channel_distribute_AADNet_Das_LOCO_acc.npy'] 
                      
datasets = ['EventAAD', 'DTU', 'KULeuven']
channel_loc_files = [f'{output_path}/eventaad_chan32.locs', f'{output_path}/BioSemi64.loc', f'{output_path}/BioSemi64.loc']
channels_SI = []
ori_acc = []
for i in range(len(distribution_files)):
    channels_SI.append(np.load(os.path.join(output_path, distribution_files[i]))[0])
    ori_acc.append(np.expand_dims(np.load(os.path.join(output_path, ori_performance_files[i]))[0], axis=-1))

for i in range(len(channels_SI)):
    (n_points, n_sbj, n_chns) = channels_SI[i].shape
    montage = read_custom_montage(channel_loc_files[i])
    pos = []
    channels = []
    for j in montage.get_positions()['ch_pos']:
        channels.append(j)
        pos.append(montage.get_positions()['ch_pos'][j])
    pos = np.array(pos)
    importance = ori_acc[i][:n_points, :n_sbj, :n_chns] - channels_SI[i]
    info = create_info(ch_names=channels[:n_chns], sfreq=1.0, ch_types='eeg')
    info.set_montage(montage)
    for n in range(n_points):
        filename = os.path.join(output_path, f"channel_distribute_{datasets[i]}_{WINDOWS[n]}s_trialZscore_latest.{output_type}")
        plt.clf()
        plt.rcParams.update({'font.size': 10})
        # plot_topomap(data=importance[n].mean(axis=0), pos=pos[:n_chns,:2], sensors=True, show=False, vlim=(-0.05, 0.01))
        EvokedArray(data=100*importance[n].mean(axis=0)[:,None], info=info).plot_topomap(times=0.0, vlim=(0, 8), scalings=1, show_names=True, show=False, cbar_fmt='%0.2f', time_format='Accuracy drop', units='percentage point', res=256, size=2)
        plt.gcf().savefig(filename, dpi=600, bbox_inches='tight')
     
        
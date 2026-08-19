import numpy as np
from src.dataset import contiguous_window_starts, split_window_indices, build_state_arrays

def test_locked_gap_safe_split_counts():
    starts=np.zeros(768,dtype=bool); starts[[0,62]]=True
    valid=contiguous_window_starts(starts,12)
    train,val,test=split_window_indices(768,12,0.70,0.15,11,valid)
    assert len(valid)==746
    assert (len(train),len(val),len(test))==(522,111,91)
    assert train[0]==0 and test[-1]==756

def test_state_arrays_preserve_trajectory_markers():
    T,N=5,3
    sim={k:np.zeros((T,N),dtype=float) for k in ['Ts','Tr','H','q']}
    sim.update({'Ta':np.zeros(T),'T_source':np.zeros(T),'alpha':np.ones(T),'Q_load':np.ones(T),'time_s':np.arange(T)*900,'x_m':np.arange(N)*1000,'trajectory_start':np.array([True,False,True,False,False])})
    sensors={'measurements':np.zeros((T,N,4)), 'masks':np.ones((T,N,4))}
    arrays=build_state_arrays(sim,sensors,{})
    assert arrays['trajectory_start'].tolist()==[True,False,True,False,False]

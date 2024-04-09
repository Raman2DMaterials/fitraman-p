# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:07:04 2023

@author: msinloz
"""


def pos_boundaries_GO():
   
        p0 = 1180.
        p0_Boundary = [1150,1220]
        
        p1 = 1360.
        p1_Boundary = [1300,1400]
        
        p2 = 1500
        p2_Boundary = [1450,1550]
        
        p3 = 1580
        p3_Boundary = [1550,1610]
        
        p4 = 1620
        p4_Boundary = [1590,1650]

        
        peakPosValues = [p0, p1, p2, p3, p4]
        peakPosBoundaries = [p0_Boundary, p1_Boundary, p2_Boundary, p3_Boundary, p4_Boundary]
                       
        FW0 = 50.
        FW0_Boundary = [0.5,500]
             
        FW1 = 50.
        FW1_Boundary = [0.5,100]
        
        FW2 = 50.
        FW2_Boundary = [0.5,500]
        
        FW3 = 50.
        FW3_Boundary = [0.5,500]
        
        FW4 = 5.
        FW4_Boundary = [0.5,200]
        

                
        peakFwhmValues = [FW0, FW1, FW2, FW3, FW4] 
        peakFwhmBoundaries = [FW0_Boundary, FW1_Boundary, FW2_Boundary, FW3_Boundary, FW4_Boundary]
                
        A0 = 3000
        A0_Boundary = [300,10000.]
        
        A1 = 3000
        A1_Boundary = [300,100000.]
        
        A2 = 1000
        A2_Boundary = [100,50000.]
        
        A3 = 10000
        A3_Boundary = [1000,250000.]
        
        A4 = 1000
        A4_Boundary = [100,50000.]
        

        
        peakAmpValues = [A0, A1, A2, A3, A4] 
        peakAmpBoundaries = [A0_Boundary, A1_Boundary, A2_Boundary, A3_Boundary, A4_Boundary]
        
        def_initialValues = peakPosValues
        for i in peakFwhmValues:
            def_initialValues.append(i)
        for i in peakAmpValues:
            def_initialValues.append(i)

        boundaries_low = []
        boundaries_high = []
        for i in peakPosBoundaries:
            boundaries_low.append(i[0])
            boundaries_high.append(i[1])
        for i in peakFwhmBoundaries:
            boundaries_low.append(i[0])
            boundaries_high.append(i[1])
        for i in peakAmpBoundaries:
            boundaries_low.append(i[0])
            boundaries_high.append(i[1])

        
        boundaries_tuple = (boundaries_low,boundaries_high)
    
        return def_initialValues, boundaries_tuple


def pos_boundaries():
   
        p0 = 1349.
        p0_Boundary = [1300,1390]
        
        p1 = 1594.
        p1_Boundary = [1500,1640]
        
        p2 = 1630
        p2_Boundary = [1620,1650]
        
        p3 = 2480
        p3_Boundary = [2400,2500]
        
        p4 = 2689
        p4_Boundary = [2610,2750]
        
        p5 = 2960
        p5_Boundary = [2890,3000]
        
        p6 = 3253
        p6_Boundary = [3100,3350]
        
        peakPosValues = [p0, p1, p2, p3, p4, p5, p6]
        peakPosBoundaries = [p0_Boundary, p1_Boundary, p2_Boundary, p3_Boundary, p4_Boundary, p5_Boundary, p6_Boundary]
                       
        FW0 = 5.
        FW0_Boundary = [0.5,150]
             
        FW1 = 30.
        FW1_Boundary = [0.5,50]
        
        FW2 = 30.
        FW2_Boundary = [0.5,80]
        
        FW3 = 30.
        FW3_Boundary = [0.5,150]
        
        FW4 = 30.
        FW4_Boundary = [0.5,200]
        
        FW5 = 30.
        FW5_Boundary = [0.5,150]
        
        FW6 = 30.
        FW6_Boundary = [0.5,150]
                
        peakFwhmValues = [FW0, FW1, FW2, FW3, FW4, FW5, FW6] 
        peakFwhmBoundaries = [FW0_Boundary, FW1_Boundary, FW2_Boundary, FW3_Boundary, FW4_Boundary, FW5_Boundary, FW6_Boundary]
                
        A0 = 100000
        A0_Boundary = [0.5,25000000.]
        
        A1 = 100000
        A1_Boundary = [0.5,25000000.]
        
        A2 = 100
        A2_Boundary = [0.05,25000000.]
        
        A3 = 100
        A3_Boundary = [0.05,25000000.]
        
        A4 = 100000
        A4_Boundary = [0.5,25000000.]
        
        A5 = 100
        A5_Boundary = [0.05,25000000.]
        
        A6 = 100
        A6_Boundary = [0.05,25000000.]
        
        peakAmpValues = [A0, A1, A2, A3, A4, A5, A6] 
        peakAmpBoundaries = [A0_Boundary, A1_Boundary, A2_Boundary, A3_Boundary, A4_Boundary, A5_Boundary, A6_Boundary]
        
        def_initialValues = peakPosValues
        for i in peakFwhmValues:
            def_initialValues.append(i)
        for i in peakAmpValues:
            def_initialValues.append(i)

        boundaries_low = []
        boundaries_high = []
        for i in peakPosBoundaries:
            boundaries_low.append(i[0])
            boundaries_high.append(i[1])
        for i in peakFwhmBoundaries:
            boundaries_low.append(i[0])
            boundaries_high.append(i[1])
        for i in peakAmpBoundaries:
            boundaries_low.append(i[0])
            boundaries_high.append(i[1])

        
        boundaries_tuple = (boundaries_low,boundaries_high)
    
        return def_initialValues, boundaries_tuple
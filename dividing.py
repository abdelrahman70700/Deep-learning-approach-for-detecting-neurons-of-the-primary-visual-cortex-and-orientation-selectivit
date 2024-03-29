import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math
import scipy.io
from PIL import Image
import streamlit as st
import cmath
from math import pi
import numpy as np
from math import exp
import matplotlib.patches as patches

class data_analysis:
    def create_folder():
        import os
        dir = os.path.join("images")
        if not os.path.exists(dir):
            os.mkdir(dir)

    #Access Fluorescence Data from the Table for one cell and all cell
    def Access_Fluorescence_Data(rawF_vector,rawF_matrix,mOrientation,mTrial):
        #Extract the raw Fluorescence Data for 1 cell
        plt.plot(rawF_vector[0:mOrientation])
        plt.title("Access Fluorescence Data from the Table for one cell")
        plt.savefig('images/fig1.jpg', bbox_inches='tight')
        st.pyplot(plt)
        plt.close()
        x = plt.imshow(rawF_matrix.iloc[0:mTrial, :], aspect='auto', interpolation='none', origin='lower')
        #Extract the raw Fluorescence Data for all cells
        plt.colorbar(x)
        plt.title("Access Fluorescence Data from the Table for all cell")
        plt.savefig('images/fig2.jpg', bbox_inches='tight')
        st.pyplot(plt)
        plt.close()

    #Computing the Change in Fluorescence Relative to Baseline of One Cell
    def relative_Fluorescence_onecell(rawF_vector,rawF_matrix):
        rawF_rounded = round(rawF_vector / 10) * 10  # Round values to nearest multiple of 10
        baseline = statistics.mode(rawF_rounded)
        DFF_vector = (rawF_vector - baseline) / baseline

        plt.plot(rawF_vector)
        plt.xlabel('Row')
        plt.ylabel('Raw F')
        plt.title('Raw fluorescence values of Cell 1')
        plt.savefig("images/fig3.jpg", bbox_inches='tight')
        st.pyplot(plt)
        plt.close()
        plt.plot(DFF_vector)
        plt.xlabel('Row')
        plt.ylabel('/DeltaF/F')
        plt.title('DF/F values of Cell 1')
        plt.savefig('images/fig4.jpg', bbox_inches='tight')
        st.pyplot(plt)
        plt.close()

    #Calculate DF/F for All Cells and normlize it for all cells
    def DFF_Compute_Normlize(rawF,rawF_matrix):
        #Generate the DFF matrix
        rawF_rounded = round(rawF_matrix / 10) * 10  # Round values to nearest multiple of 10
        baseline2 = rawF_rounded.mode(axis=0, numeric_only=True)
        baseline2 = baseline2.iloc[0]
        baseline2 = pd.DataFrame(baseline2)
        baseline2 = baseline2.T
        DFF_matrix = (rawF_matrix - baseline2.values) / baseline2.values
        DFF_matrix = pd.DataFrame(DFF_matrix)
        #Create a copy of the rawF table named DFF
        DFF = rawF
        #Overwrite the raw fluorescence data with DFF values
        DFF.iloc[:, 4:] = DFF_matrix
        #Make plots of rawF and DFF matrices
        plt.imshow(rawF_matrix, aspect='auto', interpolation='none', origin='upper')
        plt.xlabel("coulmn")
        plt.ylabel("Row")
        plt.colorbar(label='RawF Value', orientation="vertical")
        plt.title("Calculate DF/F for All Cells Without normalization")
        plt.savefig('images/fig5.jpg', bbox_inches='tight')
        st.pyplot(plt)
        plt.close()

        plt.imshow(DFF_matrix,  aspect='auto', interpolation='none', origin='upper')
        plt.xlabel("coulmn")
        plt.ylabel("Row")
        plt.colorbar(label='Column', orientation="vertical")
        plt.title("Calculate DF/F for All Cells With normalization")
        plt.savefig('images/fig6.jpg', bbox_inches='tight')
        st.pyplot(plt)
        plt.close()
        return DFF

    #Extract Dff Values for Cell1 for the two conditions 
    #calculate mean Dff Values
    def Mean_DFF_Cell1(DFF):
        # print("DFF=",DFF)
        cellOn = DFF.loc[(DFF['Cycle'] == 'on') & (DFF['Orientation'] == 90)]
        # print((DFF['Cycle'] == 'on')," ",(DFF['Orientation'] == 180))
        print(len(DFF['Orientation'] == 90))
        cellOn = cellOn['Cell1']
        # print("cellOn",cellOn)
        cellOff = DFF.loc[(DFF['Cycle'] == 'off') & (DFF['Orientation'] == 90)]
        # print((DFF['Cycle'] == 'off')," ",(DFF['Orientation'] == 180))
        cellOff = cellOff['Cell1']
        # print("celloff",cellOff)
        cellOn = pd.DataFrame(cellOn)
        cellOff = pd.DataFrame(cellOff)
        meanOn = np.mean(cellOn)
        meanOff = np.mean(cellOff)
        # print("cellOnMean_DFF_Cell1", cellOn)
        # print("cellOffMean_DFF_Cell1", cellOff)
        plt.plot(cellOn)
        plt.xlabel("Row")
        plt.ylabel("Raw F")
        plt.title('Extract Dff Values for Cell1 Orientation 90 and cycle is on')
        plt.savefig('images/fig7.jpg', bbox_inches='tight')
        st.pyplot(plt)
        plt.close()

        plt.plot(cellOff)
        plt.xlabel("Row")
        plt.ylabel("Raw F")
        plt.title('Extract Dff Values for Cell1 Orientation 90 and cycle is off')
        plt.savefig('images/fig8.jpg', bbox_inches='tight')
        st.pyplot(plt)
        plt.close()

    #Plotting an Orientation Tuning Curve for 1 cell
    def Orientation_tuning_curve(DFF,numTrials):
        # numTrials = 6
        print(numTrials)
        results = DFF.groupby(["Orientation", "Cycle"]).mean()
        results = results.iloc[:, 2]
        results = pd.DataFrame(results)
        cellOn = DFF.loc[(DFF['Cycle'] == 'on')]
        cellOff = DFF.loc[(DFF['Cycle'] == 'off')]
        resultsOn1 = cellOn.groupby("Orientation").std()
        resultsOn2 = cellOn.groupby("Orientation").mean()
        resultsOn1 = resultsOn1.iloc[:, 2]
        resultsOn2 = resultsOn2.iloc[:, 2]
        resultsOn1 = pd.DataFrame(resultsOn1)
        resultsOn2 = pd.DataFrame(resultsOn2)
        resultsOn = pd.concat([resultsOn2, resultsOn1], axis=1, ignore_index=True)
        resultsOn = resultsOn.rename(columns={0: 'mean', 1: 'std'}, inplace=False)
        resultsOff1 = cellOff.groupby("Orientation").std()
        resultsOff2 = cellOff.groupby("Orientation").mean()
        resultsOff1 = resultsOff1.iloc[:, 2]
        resultsOff2 = resultsOff2.iloc[:, 2]
        resultsOff1 = pd.DataFrame(resultsOff1)
        resultsOff2 = pd.DataFrame(resultsOff2)
        resultsOff = pd.concat([resultsOff2, resultsOff1], axis=1, ignore_index=True)
        resultsOff = resultsOff.rename(columns={0: 'mean', 1: 'std'}, inplace=False)
        meanCellOn = resultsOn.iloc[:, 0]
        meanCellOn = pd.DataFrame(meanCellOn)
        stdCellOn = resultsOn.iloc[:, 1]
        stdCellOn = pd.DataFrame(stdCellOn)
        stdErrorCellOn = stdCellOn / math.sqrt(numTrials)
        meanCellOff = resultsOff.iloc[:, 0]
        meanCellOff = pd.DataFrame(meanCellOff)
        stdCellOff = resultsOff.iloc[:, 1]
        stdCellOff = pd.DataFrame(stdCellOff)
        stdErrorCellOff = stdCellOff / math.sqrt(numTrials)
        orientation = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        orientations = pd.DataFrame(orientation)
        meanCellOfff = meanCellOff.iloc[0:1, 0:1]
        meanCellOfff = meanCellOfff['mean'].values.tolist()
        stdErrorCellOnn = stdErrorCellOn['std'].values.tolist()
        meanCellOnn = meanCellOn['mean'].values.tolist()
        # print("Orientation_tuning_curvemeancellon", meanCellOn['mean'])
        # print("Orientation_tuning_curvemeancellonlen", len(meanCellOn['mean']))
        # print("Orientation_tuning_curvemeancellonshape", meanCellOn.shape)
        line0 = plt.errorbar(orientation, meanCellOnn, yerr=stdErrorCellOnn, label="ON")
        line1 = plt.axhline(y=meanCellOfff, color='r', label="OFF")
        plt.xlabel("Orientation (deg)")
        plt.ylabel("/DeltaF/F")
        # plt.yline(meanCellOff[1],"r")
        plt.legend(handles=[line0, line1])
        plt.xlim(-30, 360)
        plt.title("Orientation Tuning Curve for 1 cell with Calculate mean on and mean Off for cell")
        plt.savefig('images/fig9.jpg', bbox_inches='tight')
        st.pyplot(plt)
        plt.close()

    ##Plotting an Orientation Tuning Curve for all cell
    def Orientation_tuning_curve_allcell(DFF):
        cellOn = DFF.loc[(DFF['Cycle'] == 'on')]
        cellOff = DFF.loc[(DFF['Cycle'] == 'off')]
        resultsOn = cellOn.groupby(cellOn['Orientation']).mean()
        resultsOff = cellOff.groupby(cellOff['Orientation']).mean()
        meanOn = resultsOn.iloc[:, 2:]
        meanOff = resultsOff.iloc[:, 2:].mean()
        meanOff = pd.DataFrame(meanOff).T
        tuningCurves = meanOn.values - meanOff.values
        tuningCurves = pd.DataFrame(tuningCurves)
        tuningCurves[tuningCurves < 0] = 0
        orientation = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        line0, = plt.plot(orientation, tuningCurves.loc[:, 0], label='cell0')
        line1, = plt.plot(orientation, tuningCurves.loc[:, 1], label='cell1')
        line2, = plt.plot(orientation, tuningCurves.loc[:, 2], label='cell2')
        line3, = plt.plot(orientation, tuningCurves.loc[:, 3], label='cell3')
        line4, = plt.plot(orientation, tuningCurves.loc[:, 4], label='cell4')
        line5, = plt.plot(orientation, tuningCurves.loc[:, 5], label='cell5')
        line6, = plt.plot(orientation, tuningCurves.loc[:, 6], label='cell6')
        plt.xlabel("Orientation (deg)")
        plt.ylabel("ON response -  OFF response")
        plt.legend(handles=[line0, line1, line2, line3, line4,line5,line6])
        plt.title("Orientation Tuning curves for all cell")
        st.pyplot(plt)
        plt.savefig('images/fig10.jpg', bbox_inches='tight')
        plt.close()
        return tuningCurves

    #Compute the orientations of all nuerons
    def preferredOrientation(tuningCurves):
        #Caclulate orientation selectivity
        tuningCurves=np.array(tuningCurves)
        orientations=list(range(0,360,30))
        orientations=np.array(orientations)
        orientationsRad = orientations*pi/180
        ab=np.exp(2*1j*orientationsRad)
        z = np.dot(ab,tuningCurves)
        OSI = abs(z)/sum(tuningCurves)
        #print(OSI)
        POrad =  0.5*np.angle(z)
        PO = POrad*180/pi
        PO[PO<0] = PO[PO<0] + 180
        # st.write(OSI)
        return OSI,PO

    #Computing Populations Statistics
    # the spatial layout of neurons is organized by the orientation selectivity
    #---------------------------------------------------------------------------------------------
    ###########################################################
    def neuroArt(img, neurons,PO, m=1):
        # Create figure and axes

        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(img)

        # Create a Rectangle patch
        # Add the patch to the Axes

        for i, x in enumerate(neurons):
            rect = patches.Rectangle((int(x[1] * m), int(x[0] * m)),
                                     int(9 * m), int(9 * m), linewidth=1, edgecolor='r', facecolor='none')
            ax.annotate(f"{round(PO[i])}",(int((x[1]+1)*m), int((x[0]+7)*m)),fontsize=8)
            ax.add_patch(rect)
        # col1,col2=st.columns([0.5,0.5])
        # with col1:
        plt.title("Population Orientation Map")
        st.pyplot(fig)
        plt.close()

    ###########################################################
    def dataPre(rawF,rawF_vector,rawF_matrix,numTrials,mOrientation,mTrial):
        maxI=np.load('imagee.npy')
        neurons=np.load('neuro2.npy')
        mult=np.load('multi_neuro.npy')+1-1
        data_analysis.create_folder()
        data_analysis.Access_Fluorescence_Data(rawF_vector,rawF_matrix,mOrientation,mTrial)
        data_analysis.relative_Fluorescence_onecell(rawF_vector,rawF_matrix)
        DFF = data_analysis.DFF_Compute_Normlize(rawF,rawF_matrix)
        data_analysis.Mean_DFF_Cell1(DFF)
        data_analysis.Orientation_tuning_curve(DFF,numTrials)
        tuningCurves=data_analysis.Orientation_tuning_curve_allcell(DFF)
        OSI,PO=data_analysis.preferredOrientation(tuningCurves)
        # Neurons_Orientations()
        data_analysis.neuroArt(maxI,neurons,PO,mult)
    ############################################################

    #---------------------------------------------------------------------------------------------
    def freq(table5HZ, f):
        for x in table5HZ:
            if x > f: return x


    def reFreq(avgIntensity,t=576, trial=6, numOrientation=12, on=4, off=2, rest=2, b="Rest"):
        m =avgIntensity.shape[0]
        f = m / t
        table5HZ = [i * 5 for i in range(1, 25)]
        F = data_analysis.freq(table5HZ, f)
        T = 1 / F  # Periodic Time
        M = round(F * m / f)
        AvgIntensity = np.array([cv2.resize(avgIntensity[:, i], (1, M)) for i in range(avgIntensity.shape[1])])
        AvgIntensity = AvgIntensity[:, :, 0].transpose()

        mTrial = int(M / trial)
        Trial = []
        for i in range(1, trial + 1):
            for j in range(mTrial):
                Trial.append(i)

        Time = [int(T * i * 10) / 10 for i in range(mTrial)] * trial

        deg = int(360 / numOrientation)
        Orientation = [i for i in range(0, 360, deg)]
        mOrientation = int(mTrial / len(Orientation))
        Orientations = []
        for x in Orientation:   Orientations += ([x] * mOrientation)
        Orientations = Orientations * trial

        FRest = ["", "off", "on"]
        FON = ["on", "", "off"]
        check = []  ###radio
        if b == "Rest":
            check = FRest
        elif b == "On":
            check = FON
        Cycle = []
        for x in check:
            if x == "":
                Cycle += ([x] * (rest * F))
            elif x == "off":
                Cycle += ([x] * (off * F))
            else:
                Cycle += ([x] * (on * F))
        Cycle = Cycle * numOrientation * trial

        return AvgIntensity, Trial, Time, Orientations, Cycle,mOrientation,mTrial

    #---------------------------------------------------------------------------------------------

    def startrun(rawF, rawF_vector, rawF_matrix,Trial,mOrientation,mTrial):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:  # To display the header text using css style
            data_analysis.dataPre(rawF, rawF_vector, rawF_matrix,Trial,mOrientation,mTrial)
    #---------------------------------------------------------------------------------------------

class final_processing:
    def analyse(self):
        logo = Image.open('index.png')
        col1, col2 = st.columns([0.8, 0.2])
        with col1:  # To display the header text using css style
            st.markdown(""" <style> .font {
                font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
                </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">If You Want To Analyse The Data...</p>', unsafe_allow_html=True)

        with col2:  # To display brand logo
            st.image(logo, width=130)
        t = st.number_input('Enter The Time Of The Video in Second', 0, 10000, 576)  # time
        trial = st.number_input('Enter The Number Of Trial', 0, 100, 6)
        numOrientation = st.number_input('Enter The Number Of Orientations', 0, 100, 12)
        on = st.number_input('Enter The Number Of On', 0, 10, 4)
        off = st.number_input('Enter The Number Of Off', 0, 10, 2)
        rest = st.number_input('Enter The Number Of Rest', 0, 10, 2)
        b = st.radio("Choose The Cycle Start", ('Rest', 'On'))
        #####################################################################

        image = np.load("data.npy")
        AvgIntensity, Trial, Time, Orientations, Cycle,mOrientation,mTrial = data_analysis.reFreq(image, t, trial, numOrientation, on,
                                                                off,
                                                                rest, b)
        d = {'Trial': Trial, 'Time': Time, 'Orientation': Orientations, 'Cycle': Cycle}
        rawF = pd.DataFrame(data=d)
        df2 = pd.DataFrame(AvgIntensity)
        names = df2.columns = [f"Cell{i}" for i in range(1, df2.shape[1] + 1)]
        rawF.loc[:, names] = df2
        rawF_vector = rawF['Cell1']
        m, n = rawF.shape
        rawF_matrix = rawF.iloc[:, 4:n]
        if st.button("◀ Start"):

            data_analysis.startrun(rawF, rawF_vector, rawF_matrix,trial,mOrientation,mTrial)

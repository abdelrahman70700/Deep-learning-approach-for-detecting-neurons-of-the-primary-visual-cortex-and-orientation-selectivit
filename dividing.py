import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math
import scipy.io
from PIL import Image
import streamlit as st
# #---------------------------------------------------------------------------------------------
# def freq(table5HZ, f):
#     for x in table5HZ:
#         if x > f: return x
#
#
# def reFreq(avgIntensity,t=576, trial=6, numOrientation=12, on=4, off=2, rest=2, b="Rest"):
#     m =avgIntensity.shape[0]
#     f = m / t
#     table5HZ = [i * 5 for i in range(1, 25)]
#     F = freq(table5HZ, f)
#     T = 1 / F  # Periodic Time
#     M = round(F * m / f)
#     AvgIntensity = np.array([cv2.resize(avgIntensity[:, i], (1, M)) for i in range(avgIntensity.shape[1])])
#     AvgIntensity = AvgIntensity[:, :, 0].transpose()
#
#     mTrial = int(M / trial)
#     Trial = []
#     for i in range(1, trial + 1):
#         for j in range(mTrial):
#             Trial.append(i)
#
#     Time = [int(T * i * 10) / 10 for i in range(mTrial)] * trial
#
#     deg = int(360 / numOrientation)
#     Orientation = [i for i in range(0, 360, deg)]
#     mOrientation = int(mTrial / len(Orientation))
#     Orientations = []
#     for x in Orientation:   Orientations += ([x] * mOrientation)
#     Orientations = Orientations * trial
#
#     FRest = ["", "off", "on"]
#     FON = ["on", "", "off"]
#     check = []  ###radio
#     if b == "Rest":
#         check = FRest
#     elif b == "On":
#         check = FON
#     Cycle = []
#     for x in check:
#         if x == "":
#             Cycle += ([x] * (rest * F))
#         elif x == "off":
#             Cycle += ([x] * (off * F))
#         else:
#             Cycle += ([x] * (on * F))
#     Cycle = Cycle * numOrientation * trial
#
#     return AvgIntensity, Trial, Time, Orientations, Cycle
#
# #---------------------------------------------------------------------------------------------

# #####################################################################
# logo = Image.open('index.png')
# col1, col2 = st.columns([0.8, 0.2])
# with col1:  # To display the header text using css style
#     st.markdown(""" <style> .font {
#         font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
#         </style> """, unsafe_allow_html=True)
#     st.markdown('<p class="font">If You Want To Analyse The Data...</p>', unsafe_allow_html=True)
#
# with col2:  # To display brand logo
#     st.image(logo, width=130)
# t = st.number_input('Enter The Time Of The Video in Second', 0, 10000, 0)  # time
# trial = st.number_input('Enter The Number Of Trial', 0, 100, 0)
# numOrientation = st.number_input('Enter The Number Of Orientations', 0, 100, 0)
# on = st.number_input('Enter The Number Of On', 0, 10, 0)
# off = st.number_input('Enter The Number Of Off', 0, 10, 0)
# rest = st.number_input('Enter The Number Of Rest', 0, 10, 0)
# b = st.radio("Choose The Cycle Start", ('Rest', 'On'))
# # #####################################################################
#
# def create_folder():
#     import os
#     dir = os.path.join("images")
#     if not os.path.exists(dir):
#         os.mkdir(dir)
#
#
# def fun1(rawF_vector,rawF_matrix):
#     plt.plot(rawF_vector[40:80])
#     plt.savefig('images/fig1.jpg', bbox_inches='tight')
#     # st.pyplot(plt)
#     plt.close()
#     x = plt.imshow(rawF_matrix.iloc[0:480, :], aspect='auto', interpolation='none', origin='lower')
#     plt.colorbar(x)
#     plt.savefig('images/fig2.jpg', bbox_inches='tight')
#     # st.pyplot(plt)
#     plt.close()
#
#
# def fun2(rawF_vector,rawF_matrix):
#     rawF_rounded = round(rawF_vector / 10) * 10  # Round values to nearest multiple of 10
#     baseline = statistics.mode(rawF_rounded)
#     DFF_vector = (rawF_vector - baseline) / baseline
#
#     plt.plot(rawF_vector)
#     plt.xlabel('Row')
#     plt.ylabel('Raw F')
#     plt.title('Raw fluorescence values of Cell 1')
#     plt.savefig("images/fig3.jpg", bbox_inches='tight')
#     # st.pyplot(plt)
#     plt.close()
#
#     plt.plot(DFF_vector)
#     plt.xlabel('Row')
#     plt.ylabel('/DeltaF/F')
#     plt.title('DF/F values of Cell 1')
#     plt.savefig('images/fig4.jpg', bbox_inches='tight')
#     # st.pyplot(plt)
#     plt.close()
#
#
# def fun3(rawF,rawF_matrix):
#     rawF_rounded = round(rawF_matrix / 10) * 10  # Round values to nearest multiple of 10
#     baseline2 = rawF_rounded.mode(axis=0, numeric_only=True)
#     baseline2 = baseline2.iloc[0]
#     baseline2 = pd.DataFrame(baseline2)
#     baseline2 = baseline2.T
#     DFF_matrix = (rawF_matrix - baseline2.values) / baseline2.values
#     DFF_matrix = pd.DataFrame(DFF_matrix)
#     DFF = rawF
#     DFF.iloc[:, 4:] = DFF_matrix
#
#     plt.imshow(rawF_matrix, extent=[0, 2500, 0, 2500])
#     plt.xlabel("coulmn")
#     plt.ylabel("Row")
#     plt.colorbar(label='RawF Value', orientation="vertical")
#     plt.savefig('images/fig5.jpg', bbox_inches='tight')
#     # st.pyplot(plt)
#     plt.close()
#
#     plt.imshow(DFF_matrix, extent=[0, 2500, 0, 2500])
#     plt.xlabel("coulmn")
#     plt.ylabel("Row")
#     plt.colorbar(label='Column', orientation="vertical")
#     plt.savefig('images/fig6.jpg', bbox_inches='tight')
#     # st.pyplot(plt)
#     plt.close()
#     # print(sum((DFF['Cycle'] == 'ON') & (DFF['Orientation'] == 180)))
#     # print(DFF.loc[(DFF['Cycle'] == 'ON') & (DFF['Orientation'] == 180)])
#     return DFF
#
#
# def fun4(DFF):
#     # print("DFF=",DFF)
#     cellOn = DFF.loc[(DFF['Cycle'] == 'on') & (DFF['Orientation'] == 180)]
#     # print((DFF['Cycle'] == 'on')," ",(DFF['Orientation'] == 180))
#     print(len(DFF['Orientation'] == 180))
#     cellOn = cellOn['Cell1']
#     # print("cellOn",cellOn)
#     cellOff = DFF.loc[(DFF['Cycle'] == 'off') & (DFF['Orientation'] == 180)]
#     # print((DFF['Cycle'] == 'off')," ",(DFF['Orientation'] == 180))
#     cellOff = cellOff['Cell1']
#     # print("celloff",cellOff)
#     cellOn = pd.DataFrame(cellOn)
#     cellOff = pd.DataFrame(cellOff)
#     meanOn = np.mean(cellOn)
#     meanOff = np.mean(cellOff)
#     # print("cellOnFun4", cellOn)
#     # print("cellOffFun4", cellOff)
#     plt.plot(cellOn)
#     plt.xlabel("Row")
#     plt.ylabel("Raw F")
#     plt.title('Raw fluorescence values of Cell 1a')
#     plt.savefig('images/fig7.jpg', bbox_inches='tight')
#     # st.pyplot(plt)
#     plt.close()
#
#     plt.plot(cellOff)
#     plt.xlabel("Row")
#     plt.ylabel("Raw F")
#     plt.title('Raw fluorescence values of Cell 1a')
#     plt.savefig('images/fig8.jpg', bbox_inches='tight')
#     # st.pyplot(plt)
#     plt.close()
#
#
# def fun5(DFF):
#     numTrials = 6
#     results = DFF.groupby(["Orientation", "Cycle"]).mean()
#     results = results.iloc[:, 2]
#     results = pd.DataFrame(results)
#     cellOn = DFF.loc[(DFF['Cycle'] == 'on')]
#     cellOff = DFF.loc[(DFF['Cycle'] == 'off')]
#     resultsOn1 = cellOn.groupby("Orientation").std()
#     resultsOn2 = cellOn.groupby("Orientation").mean()
#     resultsOn1 = resultsOn1.iloc[:, 2]
#     resultsOn2 = resultsOn2.iloc[:, 2]
#     resultsOn1 = pd.DataFrame(resultsOn1)
#     resultsOn2 = pd.DataFrame(resultsOn2)
#     resultsOn = pd.concat([resultsOn2, resultsOn1], axis=1, ignore_index=True)
#     resultsOn = resultsOn.rename(columns={0: 'mean', 1: 'std'}, inplace=False)
#     resultsOff1 = cellOff.groupby("Orientation").std()
#     resultsOff2 = cellOff.groupby("Orientation").mean()
#     resultsOff1 = resultsOff1.iloc[:, 2]
#     resultsOff2 = resultsOff2.iloc[:, 2]
#     resultsOff1 = pd.DataFrame(resultsOff1)
#     resultsOff2 = pd.DataFrame(resultsOff2)
#     resultsOff = pd.concat([resultsOff2, resultsOff1], axis=1, ignore_index=True)
#     resultsOff = resultsOff.rename(columns={0: 'mean', 1: 'std'}, inplace=False)
#     meanCellOn = resultsOn.iloc[:, 0]
#     meanCellOn = pd.DataFrame(meanCellOn)
#     stdCellOn = resultsOn.iloc[:, 1]
#     stdCellOn = pd.DataFrame(stdCellOn)
#     stdErrorCellOn = stdCellOn / math.sqrt(numTrials)
#     meanCellOff = resultsOff.iloc[:, 0]
#     meanCellOff = pd.DataFrame(meanCellOff)
#     stdCellOff = resultsOff.iloc[:, 1]
#     stdCellOff = pd.DataFrame(stdCellOff)
#     stdErrorCellOff = stdCellOff / math.sqrt(numTrials)
#     orientation = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
#     orientations = pd.DataFrame(orientation)
#     meanCellOfff = meanCellOff.iloc[0:1, 0:1]
#     meanCellOfff = meanCellOfff['mean'].values.tolist()
#     stdErrorCellOnn = stdErrorCellOn['std'].values.tolist()
#     meanCellOnn = meanCellOn['mean'].values.tolist()
#     # print("fun5meancellon", meanCellOn['mean'])
#     # print("fun5meancellonlen", len(meanCellOn['mean']))
#     # print("fun5meancellonshape", meanCellOn.shape)
#     line0 = plt.errorbar(orientation, meanCellOnn, yerr=stdErrorCellOnn, label="ON")
#     line1 = plt.axhline(y=meanCellOfff, color='r', label="OFF")
#     plt.xlabel("Orientation (deg)")
#     plt.ylabel("/DeltaF/F")
#     # plt.yline(meanCellOff[1],"r")
#     plt.legend(handles=[line0, line1])
#     plt.xlim(-30, 360)
#     plt.savefig('images/fig9.jpg', bbox_inches='tight')
#     # st.pyplot(plt)
#     plt.close()
#
#
# def fun6(DFF):
#     cellOn = DFF.loc[(DFF['Cycle'] == 'on')]
#     cellOff = DFF.loc[(DFF['Cycle'] == 'off')]
#     resultsOn = cellOn.groupby(cellOn['Orientation']).mean()
#     resultsOff = cellOff.groupby(cellOff['Orientation']).mean()
#     meanOn = resultsOn.iloc[:, 2:]
#     meanOff = resultsOff.iloc[:, 2:].mean()
#     meanOff = pd.DataFrame(meanOff).T
#     tuningCurves = meanOn.values - meanOff.values
#     tuningCurves = pd.DataFrame(tuningCurves)
#     tuningCurves[tuningCurves < 0] = 0
#     orientation = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
#     line0, = plt.plot(orientation, tuningCurves.loc[:, 0], label='cell1')
#     line1, = plt.plot(orientation, tuningCurves.loc[:, 1], label='cell2')
#     line2, = plt.plot(orientation, tuningCurves.loc[:, 2], label='cell3')
#     line3, = plt.plot(orientation, tuningCurves.loc[:, 3], label='cell4')
#     line4, = plt.plot(orientation, tuningCurves.loc[:, 4], label='cell5')
#     plt.xlabel("Orientation (deg)")
#     plt.ylabel("ON response -  OFF response")
#     plt.legend(handles=[line0, line1, line2, line3, line4])
#     plt.title("Tuning curves")
#     # st.pyplot(plt)
#     plt.savefig('images/fig10.jpg', bbox_inches='tight')
#     plt.close()
#
#
# def fun7():
#     color = [[x, y, z] for x in np.arange(0.7, 1, 0.1) for y in np.arange(0, 1, 0.1) for z in np.arange(0, 1, 0.1)]
#     mat = scipy.io.loadmat('PopMap.mat')
#     [rows, columns] = mat['tuningCurves'].shape
#     # print(mat['ROIs'])
#     numCells = columns
#     fig, ax = plt.subplots()
#     x = plt.imshow(mat['img'])
#
#     colormap = plt.cm.get_cmap('hsv')
#     sm = plt.cm.ScalarMappable(cmap=colormap)
#     sm.set_clim(vmin=0, vmax=180)
#
#     for i in range(0, columns):
#         roi = mat['ROIs'][0][i][0][0][4]
#         if np.isnan(mat['OSI'][0][i]):
#             roiColor = np.zeros(3)
#         # plt.scatter(mat['ROIs'][0][i][0][0][4][:, 0], mat['ROIs'][0][i][0][0][4][:, 1], color=roiColor)
#         elif mat['OSI'][0][i] < 0.25:
#             roiColor = np.ones(3)
#         # plt.scatter(mat['ROIs'][0][i][0][0][4][:, 0],mat['ROIs'][0][i][0][0][4][:, 1],color=roiColor)
#
#         else:
#             c = math.ceil(mat['PO'][0][i])
#             roiColor = color[c]
#             # print(roiColor)
#         sc = plt.scatter(mat['ROIs'][0][i][0][0][4][:, 0], mat['ROIs'][0][i][0][0][4][:, 1], color=(roiColor), s=4)
#     plt.colorbar(sm, label="Preferred orientation (deg)", orientation="vertical")
#     plt.title("Population Orientation Map")
#     # plt.show()
#     plt.savefig('images/fig11.jpg', bbox_inches='tight')
#     # st.pyplot(plt)
#     plt.close()
# #---------------------------------------------------------------------------------------------
#
# # fun3()
# # def Data_preprocessing():
# #     create_folder()
# #     fun1()
# #     fun2()
# #     DFF=fun3()
# #     fun4(DFF)
# #     fun5(DFF)
# #     fun6(DFF)
# #     fun7()
# ###########################################################
# def dataPre(rawF,rawF_vector,rawF_matrix):
#     create_folder()
#     fun1(rawF_vector,rawF_matrix)
#     fun2(rawF_vector,rawF_matrix)
#     DFF = fun3(rawF,rawF_matrix)
#     fun4(DFF)
#     fun5(DFF)
#     fun6(DFF)
#     fun7()
#
# ############################################################
###########################################################
# image=np.load("data.npy")
# AvgIntensity, Trial, Time, Orientations, Cycle = reFreq(image, t, trial, numOrientation, on,
#                                                         off,
#                                                         rest, b)
# d = {'Trial': Trial, 'Time': Time, 'Orientation': Orientations, 'Cycle': Cycle}
# rawF = pd.DataFrame(data=d)
# # print("df1shape",rawF.shape)
# df2 = pd.DataFrame(AvgIntensity)
# # print("df2shape",df2.shape)
# names = df2.columns = [f"Cell{i}" for i in range(1, df2.shape[1] + 1)]
# rawF.loc[:, names] = df2
# rawF_vector = rawF['Cell1']
# m, n = rawF.shape
# rawF_matrix = rawF.iloc[:, 4:n]
# dataPre(rawF, rawF_vector, rawF_matrix)
############################################################
# def Data_preprocessing1():
#     create_folder()
#     fun4()
#     fun5()
#     fun6()
#     fun7()

# Data_preprocessing()


# #####################################################################

def create_folder():
    import os
    dir = os.path.join("images")
    if not os.path.exists(dir):
        os.mkdir(dir)


def fun1(rawF_vector,rawF_matrix):
    plt.plot(rawF_vector[40:80])
    plt.savefig('images/fig1.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()
    x = plt.imshow(rawF_matrix.iloc[0:480, :], aspect='auto', interpolation='none', origin='lower')
    plt.colorbar(x)
    plt.savefig('images/fig2.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()


def fun2(rawF_vector,rawF_matrix):
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


def fun3(rawF,rawF_matrix):
    rawF_rounded = round(rawF_matrix / 10) * 10  # Round values to nearest multiple of 10
    baseline2 = rawF_rounded.mode(axis=0, numeric_only=True)
    baseline2 = baseline2.iloc[0]
    baseline2 = pd.DataFrame(baseline2)
    baseline2 = baseline2.T
    DFF_matrix = (rawF_matrix - baseline2.values) / baseline2.values
    DFF_matrix = pd.DataFrame(DFF_matrix)
    DFF = rawF
    DFF.iloc[:, 4:] = DFF_matrix

    plt.imshow(rawF_matrix, extent=[0, 2500, 0, 2500])
    plt.xlabel("coulmn")
    plt.ylabel("Row")
    plt.colorbar(label='RawF Value', orientation="vertical")
    plt.savefig('images/fig5.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()

    plt.imshow(DFF_matrix, extent=[0, 2500, 0, 2500])
    plt.xlabel("coulmn")
    plt.ylabel("Row")
    plt.colorbar(label='Column', orientation="vertical")
    plt.savefig('images/fig6.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()
    # print(sum((DFF['Cycle'] == 'ON') & (DFF['Orientation'] == 180)))
    # print(DFF.loc[(DFF['Cycle'] == 'ON') & (DFF['Orientation'] == 180)])
    return DFF


def fun4(DFF):
    # print("DFF=",DFF)
    cellOn = DFF.loc[(DFF['Cycle'] == 'on') & (DFF['Orientation'] == 180)]
    # print((DFF['Cycle'] == 'on')," ",(DFF['Orientation'] == 180))
    print(len(DFF['Orientation'] == 180))
    cellOn = cellOn['Cell1']
    # print("cellOn",cellOn)
    cellOff = DFF.loc[(DFF['Cycle'] == 'off') & (DFF['Orientation'] == 180)]
    # print((DFF['Cycle'] == 'off')," ",(DFF['Orientation'] == 180))
    cellOff = cellOff['Cell1']
    # print("celloff",cellOff)
    cellOn = pd.DataFrame(cellOn)
    cellOff = pd.DataFrame(cellOff)
    meanOn = np.mean(cellOn)
    meanOff = np.mean(cellOff)
    # print("cellOnFun4", cellOn)
    # print("cellOffFun4", cellOff)
    plt.plot(cellOn)
    plt.xlabel("Row")
    plt.ylabel("Raw F")
    plt.title('Raw fluorescence values of Cell 1a')
    plt.savefig('images/fig7.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()

    plt.plot(cellOff)
    plt.xlabel("Row")
    plt.ylabel("Raw F")
    plt.title('Raw fluorescence values of Cell 1a')
    plt.savefig('images/fig8.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()


def fun5(DFF):
    numTrials = 6
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
    # print("fun5meancellon", meanCellOn['mean'])
    # print("fun5meancellonlen", len(meanCellOn['mean']))
    # print("fun5meancellonshape", meanCellOn.shape)
    line0 = plt.errorbar(orientation, meanCellOnn, yerr=stdErrorCellOnn, label="ON")
    line1 = plt.axhline(y=meanCellOfff, color='r', label="OFF")
    plt.xlabel("Orientation (deg)")
    plt.ylabel("/DeltaF/F")
    # plt.yline(meanCellOff[1],"r")
    plt.legend(handles=[line0, line1])
    plt.xlim(-30, 360)
    plt.savefig('images/fig9.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()


def fun6(DFF):
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
    line0, = plt.plot(orientation, tuningCurves.loc[:, 0], label='cell1')
    line1, = plt.plot(orientation, tuningCurves.loc[:, 1], label='cell2')
    line2, = plt.plot(orientation, tuningCurves.loc[:, 2], label='cell3')
    line3, = plt.plot(orientation, tuningCurves.loc[:, 3], label='cell4')
    line4, = plt.plot(orientation, tuningCurves.loc[:, 4], label='cell5')
    plt.xlabel("Orientation (deg)")
    plt.ylabel("ON response -  OFF response")
    plt.legend(handles=[line0, line1, line2, line3, line4])
    plt.title("Tuning curves")
    st.pyplot(plt)
    plt.savefig('images/fig10.jpg', bbox_inches='tight')
    plt.close()


def fun7():
    color = [[x, y, z] for x in np.arange(0.7, 1, 0.1) for y in np.arange(0, 1, 0.1) for z in np.arange(0, 1, 0.1)]
    mat = scipy.io.loadmat('PopMap.mat')
    [rows, columns] = mat['tuningCurves'].shape
    # print(mat['ROIs'])
    numCells = columns
    fig, ax = plt.subplots()
    x = plt.imshow(mat['img'])

    colormap = plt.cm.get_cmap('hsv')
    sm = plt.cm.ScalarMappable(cmap=colormap)
    sm.set_clim(vmin=0, vmax=180)

    for i in range(0, columns):
        roi = mat['ROIs'][0][i][0][0][4]
        if np.isnan(mat['OSI'][0][i]):
            roiColor = np.zeros(3)
        # plt.scatter(mat['ROIs'][0][i][0][0][4][:, 0], mat['ROIs'][0][i][0][0][4][:, 1], color=roiColor)
        elif mat['OSI'][0][i] < 0.25:
            roiColor = np.ones(3)
        # plt.scatter(mat['ROIs'][0][i][0][0][4][:, 0],mat['ROIs'][0][i][0][0][4][:, 1],color=roiColor)

        else:
            c = math.ceil(mat['PO'][0][i])
            roiColor = color[c]
            # print(roiColor)
        sc = plt.scatter(mat['ROIs'][0][i][0][0][4][:, 0], mat['ROIs'][0][i][0][0][4][:, 1], color=(roiColor), s=4)
    plt.colorbar(sm, label="Preferred orientation (deg)", orientation="vertical")
    plt.title("Population Orientation Map")
    # plt.show()
    plt.savefig('images/fig11.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()
#---------------------------------------------------------------------------------------------
###########################################################
def dataPre(rawF,rawF_vector,rawF_matrix):
    create_folder()
    fun1(rawF_vector,rawF_matrix)
    fun2(rawF_vector,rawF_matrix)
    DFF = fun3(rawF,rawF_matrix)
    fun4(DFF)
    fun5(DFF)
    fun6(DFF)
    fun7()
############################################################

#---------------------------------------------------------------------------------------------
def freq(table5HZ, f):
    for x in table5HZ:
        if x > f: return x


def reFreq(avgIntensity,t=576, trial=6, numOrientation=12, on=4, off=2, rest=2, b="Rest"):
    m =avgIntensity.shape[0]
    f = m / t
    table5HZ = [i * 5 for i in range(1, 25)]
    F = freq(table5HZ, f)
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

    return AvgIntensity, Trial, Time, Orientations, Cycle

#---------------------------------------------------------------------------------------------

def startrun(rawF, rawF_vector, rawF_matrix):
    col1, col2 = st.columns([0.8, 0.2])
    with col1:  # To display the header text using css style
        dataPre(rawF, rawF_vector, rawF_matrix)


def analyse():
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
    AvgIntensity, Trial, Time, Orientations, Cycle = reFreq(image, t, trial, numOrientation, on,
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
        startrun(rawF, rawF_vector, rawF_matrix)

from email.header import Header
import os as os
import csv as csv
import zipfile as zp
import math as ma
from numpy import indices
import pyperclip as pyclip
import copy as cp
###############
# import matplotlib.pyplot as plt
import pathlib
###############


def PrintMatrix(matrix):
    for row in matrix:
        print(row)


def SortPerIndex(OriginalData, Index=0, Reverse=True):
    Data = cp.deepcopy(OriginalData)
    try:
        Data.sort(key=lambda x: x[Index])
        if Reverse:
            Data.reverse()
        return Data
    except:
        raise ValueError(
            "El índice dado excede el máximo posible en la matriz, o la matriz aportada no es adecuada.")


def Threshold(InputData, Percentage):
    ThresholdData = cp.deepcopy(InputData)
    RowNumber = len(InputData)
    ColNumber = len(InputData[0])
    TotalVotes = [sum([InputData[i][j] for j in range(ColNumber)])
                  for i in range(RowNumber)]
    for i in range(RowNumber):
        for j in range(ColNumber):
            if InputData[i][j]/TotalVotes[i] < Percentage:
                ThresholdData[i][j] = 0
    return ThresholdData

########################
# Indices de desproporc.


def LoosemoreHanbyIndex(Votes, Seats):
    DiffList = []
    for (vote, seat) in zip(Votes, Seats):
        DiffList += [round(abs(vote - seat), 2)]
    return (round(sum(DiffList)/2, 2), DiffList)


def RaeIndex(Votes, Seats):
    DiffList = []
    cont = 0
    for (vote, seat) in zip(Votes, Seats):
        if vote > 0.5:
            DiffList += [round(abs(vote - seat), 2)]
            cont += 1
        else:
            DiffList += [0]
    return (round(sum(DiffList)/cont, 2), DiffList)


def LeastSquareIndex(Votes, Seats, Rounding=2):
    DiffList = []
    for (vote, seat) in zip(Votes, Seats):
        DiffList += [round((vote - seat)**2, Rounding)]
    return (round((sum(DiffList)/2)**(1/2), Rounding), DiffList)


def SainteLagueIndex(Votes, Seats, Rounding=2):
    DiffList = []
    for (vote, seat) in zip(Votes, Seats):
        if seat > 0 and vote > 0:
            DiffList += [round((vote - seat)**2/vote, Rounding)]
        else:
            DiffList += [0]
    return (round(sum(DiffList), Rounding), DiffList)


def JeffersonIndex(Votes, Seats, Percentage=0, Rounding=2):
    DiffList = []
    for (vote, seat) in zip(Votes, Seats):
        if seat > Percentage and vote > 0:
            DiffList += [round(seat/vote, Rounding)]
        else:
            DiffList += [0]
    return (max(DiffList), DiffList)

################################


def LatexTableColumnIndexGenerator(ColNumber):
    Output = ""
    aux = ['|c|']+['r' for i in range(ColNumber-1)]+['|']
    return Output.join(aux)


def RoundPosMatrix(Matrix, Decimal=2):
    if not(isinstance(Matrix, list)) or not(all(isinstance(element, list) for element in Matrix)) or not(all(all(type(number) in [float, int] and number >= 0 for number in element) for element in Matrix)) or not(all(len(element) == len(Matrix[0]) for element in Matrix)):
        raise TypeError(
            "First argument must be a list of lists of equal length composed of nonnegative numbers.")
    return [[round(Matrix[i][j], Decimal) for j in range(len(Matrix[i]))] for i in range(len(Matrix))]


def WebsterSignpost(Num):
    if not(type(Num) in [int, float]):
        raise TypeError("Argument must be a nonnegative number.")
    if Num < 0:
        raise ValueError("Argument must be nonnegative.")
    if Num == 0:
        return 0
    else:
        return round(Num-0.5, 1)


def WebsterSeatAllocation(Num, Style="M"):
    """
    Style indicates if given ties it should take the higher possible\n
    value or the smaller value and indicate with a \" + 0.01\"

    >>> WebsterSeatAllocation(3.2,"M")

    Parameters
    ----------
    Num : int
        number for blah blah

    Returns
    -----------
    A number
    """
    if not(type(Num) in [int, float]):
        raise TypeError("First argument must be a nonnegative number.")
    if Num < 0:
        raise ValueError("First argument must be nonnegative.")
    PossibleStyles = ["M", "I", "m"]
    if not(isinstance(Style, str)) or Style not in PossibleStyles:
        raise TypeError(
            "Second argument must be a string contained in " + str(PossibleStyles))
    if Num - ma.floor(Num) > 0.5:
        return ma.ceil(Num)
    elif Num - ma.floor(Num) < 0.5:
        return ma.floor(Num)
    else:
        if Style == "M":
            return ma.ceil(Num)
        elif Style == "I":
            return(round(ma.floor(Num) + 0.1, 2))
        elif Style == "m":
            return(ma.floor(Num))


def Jefferson(VoteList, HouseSize):
    if not(isinstance(VoteList, list)):
        raise TypeError(
            "First argument must be a list containing nonnegative integers.")
    if not(all(isinstance(votes, int) and votes >= 0 for votes in VoteList)):
        raise ValueError(
            "First argument must a list containing nonnegative integers.")
    if not(isinstance(HouseSize, int)):
        raise TypeError("Second argument must be a nonnegative integer.")
    if HouseSize < 0:
        raise ValueError("Second argument must be a nonnegative integer.")
    TotalVotes = sum(VoteList)
    WeightList = [votes*HouseSize/TotalVotes for votes in VoteList]
    SeatList = [ma.floor(weight) for weight in WeightList]

    while sum(SeatList) < HouseSize:
        factor = min([ma.floor(SeatList[i]+1)/WeightList[i]
                      for i in range(len(VoteList)) if WeightList[i] != 0])
        WeightList = [weight*factor for weight in WeightList]
        SeatList = [ma.floor(weight) for weight in WeightList]
    while sum(SeatList) > HouseSize:  # If ties within the updated weights
        index = [index for index in range(len(VoteList)) if int(
            WeightList[index]) == WeightList[index]][0]
        SeatList[index] += -1
        WeightList[index] += -0.5
    return SeatList


def Webster(VoteList, HouseSize):
    if not(isinstance(VoteList, list)):
        raise TypeError(
            "First argument must be a list containing nonnegative integers.")
    if not(all(type(votes) in [int, float] and votes >= 0 for votes in VoteList)):
        raise ValueError(
            "First argument must a list containing nonnegative integers.")
    if not(isinstance(HouseSize, int)):
        raise TypeError("Second argument must be a nonnegative integer.")
    if HouseSize < 0:
        raise ValueError("Second argument must be a nonnegative integer.")
    if all(votes == 0 for votes in VoteList):  # cuidao
        return ([0 for votes in VoteList], [0 for votes in VoteList])
    TotalVotes = sum(VoteList)
    WeightList = [round(votes*HouseSize/TotalVotes, 2)
                  for votes in VoteList]
    MaxSeatList = [WebsterSeatAllocation(weight) for weight in WeightList]
    MinSeatList = [WebsterSeatAllocation(weight, "m") for weight in WeightList]
    if sum(MaxSeatList) < HouseSize:
        while sum(MaxSeatList) < HouseSize:
            factor = min([WebsterSignpost(MaxSeatList[i]+1)/WeightList[i]
                          for i in range(len(VoteList)) if WeightList[i] != 0])
            WeightList = [round(weight*factor, 2) for weight in WeightList]
            MaxSeatList = [WebsterSeatAllocation(
                weight) for weight in WeightList]
        SeatList = [WebsterSeatAllocation(weight, "I")
                    for weight in WeightList]
        while sum([ma.floor(seat) for seat in SeatList]) < HouseSize:
            SeatList[[index for index in range(len(SeatList)) if isinstance(SeatList[index], float) and round(SeatList[index]-ma.floor(
                SeatList[index]), 2) == 0.1][0]] = round(SeatList[[index for index in range(len(SeatList)) if isinstance(SeatList[index], float) and round(SeatList[index]-ma.floor(
                    SeatList[index]), 2) == 0.1][0]]+0.91, 2)
        SeatList = [round(seat, 2) for seat in SeatList]
        WeightList = [round(weight, 2) for weight in WeightList]

        return (SeatList, WeightList)
    if sum(MinSeatList) > HouseSize:
        while sum(MinSeatList) > HouseSize:
            factor = max([WebsterSignpost(MinSeatList[i])/WeightList[i]
                          for i in range(len(VoteList)) if WeightList[i] != 0])
            WeightList = [round(weight*factor, 2) for weight in WeightList]
            MinSeatList = [WebsterSeatAllocation(
                weight, "m") for weight in WeightList]
        SeatList = [WebsterSeatAllocation(weight, "I")
                    for weight in WeightList]
        while sum([ma.floor(seat) for seat in SeatList]) < HouseSize:
            SeatList[[index for index in range(len(SeatList)) if isinstance(SeatList[index], float) and round(SeatList[index]-ma.floor(
                SeatList[index]), 2) == 0.1][0]] = round(SeatList[[index for index in range(len(SeatList)) if isinstance(SeatList[index], float) and round(SeatList[index]-ma.floor(
                    SeatList[index]), 2) == 0.1][0]]+0.91, 2)
        SeatList = [round(seat, 2) for seat in SeatList]
        WeightList = [round(weight, 2) for weight in WeightList]

        return (SeatList, WeightList)
    else:
        SeatList = [WebsterSeatAllocation(weight, "I")
                    for weight in WeightList]
        while sum([ma.floor(seat) for seat in SeatList]) < HouseSize:
            SeatList[[index for index in range(len(SeatList)) if isinstance(SeatList[index], float) and round(SeatList[index]-ma.floor(
                SeatList[index]), 2) == 0.1][0]] = round(SeatList[[index for index in range(len(SeatList)) if isinstance(SeatList[index], float) and round(SeatList[index]-ma.floor(
                    SeatList[index]), 2) == 0.1][0]]+0.91, 2)
        SeatList = [round(seat, 2) for seat in SeatList]
        WeightList = [round(weight, 2) for weight in WeightList]

        return [SeatList, WeightList]


def DirectWebster(WeightList, HouseSize):
    if not(isinstance(WeightList, list)):
        raise TypeError(
            "First argument must be a list containing nonnegative integers.")
    if not(all(type(weight) in [int, float] and weight >= 0 for weight in WeightList)):
        raise ValueError(
            "First argument must a list containing nonnegative integers.")
    if not(isinstance(HouseSize, int)):
        raise TypeError("Second argument must be a nonnegative integer.")
    if HouseSize < 0:
        raise ValueError("Second argument must be a nonnegative integer.")
    SeatList = [WebsterSeatAllocation(weight, "I")
                for weight in WeightList]
    while sum([ma.floor(seat) for seat in SeatList]) < HouseSize:
        SeatList[[index for index in range(len(SeatList)) if isinstance(SeatList[index], float) and round(SeatList[index]-ma.floor(
            SeatList[index]), 2) == 0.1][0]] = round(SeatList[[index for index in range(len(SeatList)) if isinstance(SeatList[index], float) and round(SeatList[index]-ma.floor(
                SeatList[index]), 2) == 0.1][0]]+0.91, 2)
    SeatList = [round(seat, 2) for seat in SeatList]

    return SeatList


def ColumnFinder(List, ForbiddenColumns):
    ColumnOptions = [index for index in range(len(List)) if not(
        index in ForbiddenColumns) and List[index] == "D"]
    if len(ColumnOptions) > 0:
        return ColumnOptions
    else:
        return "E"


def RowFinder(List, ForbiddenRows):
    RowOptions = [index for index in range(len(List)) if not(
        index in ForbiddenRows) and List[index] == "I"]
    if len(RowOptions) > 0:
        return RowOptions
    else:
        return "E"


def PathFinder(IncrDecrMatrix, StartRows, EndRows, Path=[], StartColumns=[], RowColumn="C"):
    if any(row in [coordinates[0] for coordinates in Path] for row in EndRows):
        return Path
    elif len(StartRows) == len(IncrDecrMatrix) or len(StartColumns) == len(IncrDecrMatrix[0]):
        return "NP"
    else:
        if len(Path) == 0:
            for row in StartRows:
                Columns = ColumnFinder(IncrDecrMatrix[row], StartColumns)
                if Columns != "E":
                    for col in Columns:
                        if col not in StartColumns:
                            AuxPath = Path + [[row, col]]
                            AuxStartColumns = StartColumns + [col]
                            PossibleOutput = PathFinder(
                                IncrDecrMatrix, StartRows, EndRows, AuxPath, AuxStartColumns, "R")
                            if PossibleOutput == "NP":
                                next
                            else:
                                return PossibleOutput
            return "NP"
        else:
            if RowColumn == "C":
                row = Path[-1][0]
                Columns = ColumnFinder(IncrDecrMatrix[row], StartColumns)
                if Columns != "E":
                    for col in Columns:
                        if col not in StartColumns:
                            AuxPath = Path + [[row, col]]
                            AuxStartColumns = StartColumns + [col]
                            PossibleOutput = PathFinder(
                                IncrDecrMatrix, StartRows, EndRows, AuxPath, AuxStartColumns, "R")
                            if PossibleOutput == "NP":
                                next
                            else:
                                return PossibleOutput
                return "NP"
            if RowColumn == "R":
                col = Path[-1][1]
                Rows = RowFinder([IncrDecrMatrix[row][col]
                                  for row in range(len(IncrDecrMatrix))], StartRows)
                if Rows != "E":
                    for row in Rows:
                        if row not in StartRows:
                            AuxPath = Path + [[row, col]]
                            AuxStartRows = StartRows + [row]
                            PossibleOutput = PathFinder(
                                IncrDecrMatrix, AuxStartRows, EndRows, AuxPath, StartColumns, "C")
                            if PossibleOutput == "NP":
                                next
                            else:
                                return PossibleOutput
                return "NP"
        return "NP"


def UpdateLabeled(IncDecMatrix, LabeledRows, LabeledColumns=[]):
    LRSize = len(LabeledRows)
    LCSize = len(LabeledColumns)
    for row in LabeledRows:
        NewLabeledColumns = ColumnFinder(IncDecMatrix[row], LabeledColumns)
        if NewLabeledColumns != "E":
            LabeledColumns += NewLabeledColumns
    for col in LabeledColumns:
        NewLabeledRows = RowFinder([IncDecMatrix[index][col]
                                    for index in range(len(IncDecMatrix))], LabeledRows)
        if NewLabeledRows != "E":
            LabeledRows += NewLabeledRows
    if LCSize == len(LabeledColumns) and LRSize == len(LabeledRows):
        return (LabeledRows, LabeledColumns)
    else:
        return UpdateLabeled(IncDecMatrix, LabeledRows, LabeledColumns)


def TieTransfer(VMatrix):
    if not(isinstance(VMatrix, WeightSeatMatrix)):
        raise TypeError("Argument given must be from VoteWeightMatrix class.")
    if len(VMatrix.SeatMatrix) == 0:
        VMatrix.UpdateSeatMatrix()
        VMatrix.UpdateIncrDecrMatrix()
    if len(VMatrix.IncrDecrMatrix) == 0:
        VMatrix.UpdateIncrDecrMatrix()

    OverRepresentation = sum([sum(VMatrix.SeatMatrix[row]) - VMatrix.DistrictSeats[row]
                              for row in VMatrix.OverRepresentedDistricts])
    UnderRepresentation = sum([VMatrix.DistrictSeats[row] - sum(VMatrix.SeatMatrix[row])
                               for row in VMatrix.UnderRepresentedDistricts])
    FlawCount = OverRepresentation + UnderRepresentation
    if FlawCount == 0:
        return None
    Path = PathFinder(VMatrix.IncrDecrMatrix,
                      VMatrix.OverRepresentedDistricts, VMatrix.UnderRepresentedDistricts)
    if Path != "NP":
        for index in Path:
            seat = VMatrix.SeatMatrix[index[0]][index[1]]
            if seat - ma.floor(seat) > 0.02:
                VMatrix.SeatMatrix[index[0]][index[1]] = round(seat + 0.91, 2)
            else:
                VMatrix.SeatMatrix[index[0]][index[1]] = round(seat - 0.91, 2)
        VMatrix.UpdateSeatMatrix()
        VMatrix.UpdateIncrDecrMatrix()
        FlawCount += -2
        if FlawCount == 0:
            return None
        else:
            return TieTransfer(VMatrix)
    LabeledRows = [VMatrix.OverRepresentedDistricts[0]]
    LabeledColumns = []
    (LabeledRows, LabeledColumns) = UpdateLabeled(
        VMatrix.IncrDecrMatrix, LabeledRows, LabeledColumns)
    while OverRepresentation + UnderRepresentation == FlawCount:
        updat = VMatrix.SearchUpdateFactor(LabeledRows, LabeledColumns)
        VMatrix.RowColumnDivisors(LabeledRows, 1/updat, LabeledColumns, updat)
        VMatrix.UpdateSeatMatrix()
        VMatrix.UpdateIncrDecrMatrix()
        Path = PathFinder(VMatrix.IncrDecrMatrix,
                          VMatrix.OverRepresentedDistricts, VMatrix.UnderRepresentedDistricts)
        if Path != "NP":
            for index in Path:
                seat = VMatrix.SeatMatrix[index[0]][index[1]]
                if seat - ma.floor(seat) > 0.02:
                    VMatrix.SeatMatrix[index[0]][index[1]
                                                 ] = round(seat + 0.91, 2)
                else:
                    VMatrix.SeatMatrix[index[0]][index[1]
                                                 ] = round(seat - 0.91, 2)
            VMatrix.UpdateSeatMatrix()
            VMatrix.UpdateIncrDecrMatrix()
            FlawCount += -2
            if FlawCount > 0:
                return TieTransfer(VMatrix)
            else:
                return None
        (LabeledRows, LabeledColumns) = UpdateLabeled(
            VMatrix.IncrDecrMatrix, LabeledRows, LabeledColumns)
    return None

###################################################
#             WEIGHTSEATMATRIX CLASS              #
###################################################


class WeightSeatMatrix:
    """
    WeightSeat Matrix
    >>> 
    """

    def __init__(self, VoteMatrix, DistrictNames, PartyNames, DistrictSizes):
        if not(isinstance(VoteMatrix, list)) or not(all(isinstance(element, list) for element in VoteMatrix)) or not(all(all(type(number) in [float, int] and number >= 0 for number in element) for element in VoteMatrix)) or not(all(len(element) == len(VoteMatrix[0]) for element in VoteMatrix)):
            raise TypeError(
                "First argument must be a list of lists of equal length composed of nonnegative numbers.")
        if not(isinstance(DistrictNames, list)) or not(all(isinstance(district, str) for district in DistrictNames)) or len(DistrictNames) != len(VoteMatrix):
            raise TypeError(
                "Second argument must consist of a list of str names with the same length as the number of lists given within the first argument.")
        if not(isinstance(PartyNames, list)) or not(all(isinstance(party, str) for party in PartyNames)) or len(PartyNames) != len(VoteMatrix[0]):
            raise TypeError(
                "Third argument must consist of a list of str names with the same length as the numbers of elements of every list given within the first argument.")
        if not(isinstance(DistrictSizes, list)) or len(DistrictSizes) != len(VoteMatrix):
            raise TypeError(
                "Fourth argument must be a list containing positive integers with same length as the first argument.")
        if not(all(isinstance(size, int) and size > 0 for size in DistrictSizes)):
            raise ValueError(
                "Fifth argument must contain positive integers with sum equal to the house size.")
        self.VoteMatrix = VoteMatrix
        self.HouseSize = sum(DistrictSizes)
        self.TotalPartyVotes = [sum(row[j] for row in self.VoteMatrix)
                                for j in range(len(self.VoteMatrix[0]))]
        self.TotalPartySeats = Jefferson(self.TotalPartyVotes, self.HouseSize)
        self.DistrictSeats = DistrictSizes
        self.TotalVotes = sum(self.TotalPartyVotes)
        self.Districts = DistrictNames
        self.DistrictsNumber = len(DistrictNames)
        self.Parties = PartyNames
        self.PartiesNumber = len(PartyNames)
        self.WeightMatrix = RoundPosMatrix([[self.VoteMatrix[i][j]*self.HouseSize/self.TotalVotes for j in range(
            self.PartiesNumber)] for i in range(self.DistrictsNumber)])
        self.SeatMatrix = []
        self.IncrDecrMatrix = []
        self.OverRepresentedDistricts = []
        self.UnderRepresentedDistricts = []

    def RowColumnDivisors(self, RowList=[], RowDivisor=1, ColList=[], ColDivisor=1):
        """
        Updates Weight Matrix
        """
        if not(isinstance(RowList, list)) or not(all(isinstance(index, int) and index < self.DistrictsNumber for index in RowList)):
            raise TypeError("Row indices must be inferior than " +
                            str(self.DistrictsNumber)+" and must be provided inside a list.")
        if not(type(RowDivisor) in [int, float]):
            raise TypeError("Row divisor if given must be positive number.")
        if not(RowDivisor > 0):
            raise ValueError("Row divisor must be positive.")
        for index in RowList:
            for j in range(self.PartiesNumber):
                self.WeightMatrix[index][j] *= 1/RowDivisor
        if not(isinstance(ColList, list)) or not(all(isinstance(index, int) and index < self.PartiesNumber for index in ColList)):
            raise TypeError("Column indices must be inferior than " +
                            str(self.PartiesNumber)+" and must be provided inside a list.")
        if not(type(ColDivisor) in [int, float]):
            raise TypeError("Column divisor if given must be positive number.")
        if not(ColDivisor > 0):
            raise ValueError("Column divisor must be positive.")
        for i in range(self.DistrictsNumber):
            for index in ColList:
                self.WeightMatrix[i][index] *= 1/ColDivisor
        self.WeightMatrix = RoundPosMatrix(self.WeightMatrix)

    def UpdateSeatMatrix(self):
        if len(self.SeatMatrix) == 0:
            AuxWebsterVector = [Webster([self.WeightMatrix[i][j] for i in range(
                self.DistrictsNumber)], self.TotalPartySeats[j]) for j in range(self.PartiesNumber)]
            AuxSeatMatrix = [[AuxWebsterVector[j][0][i] for j in range(
                self.PartiesNumber)] for i in range(self.DistrictsNumber)]
            self.WeightMatrix = RoundPosMatrix([[AuxWebsterVector[j][1][i] for j in range(self.PartiesNumber)] for i in range(
                self.DistrictsNumber)])
            self.SeatMatrix = AuxSeatMatrix
            # Corregir
            self.UnderRepresentedDistricts = [index for index in range(self.DistrictsNumber) if sum(
                [int(self.SeatMatrix[index][j]) for j in range(self.PartiesNumber)]) < self.DistrictSeats[index]]
            self.OverRepresentedDistricts = [index for index in range(self.DistrictsNumber) if sum(
                [int(self.SeatMatrix[index][j]) for j in range(self.PartiesNumber)]) > self.DistrictSeats[index]]
        else:
            AuxSeatMatrix = [DirectWebster([self.WeightMatrix[i][j] for i in range(
                self.DistrictsNumber)], self.TotalPartySeats[j]) for j in range(self.PartiesNumber)]
            AuxSeatMatrix = [list(i) for i in zip(
                *AuxSeatMatrix)]  # transpose matrix
            for i in range(self.DistrictsNumber):
                for j in range(self.PartiesNumber):
                    # creacion empates
                    if isinstance(AuxSeatMatrix[i][j], float):
                        if ma.floor(AuxSeatMatrix[i][j]) != ma.floor(self.SeatMatrix[i][j]):
                            if round(AuxSeatMatrix[i][j]-ma.floor(AuxSeatMatrix[i][j]), 2) == 0.01:
                                AuxSeatMatrix[i][j] = round(
                                    AuxSeatMatrix[i][j]-0.91, 2)
                            elif round(AuxSeatMatrix[i][j]-ma.floor(AuxSeatMatrix[i][j]), 2) == 0.1:
                                AuxSeatMatrix[i][j] = round(
                                    AuxSeatMatrix[i][j]+0.91, 2)
            self.SeatMatrix = AuxSeatMatrix
            self.UnderRepresentedDistricts = [index for index in range(self.DistrictsNumber) if sum(
                [int(self.SeatMatrix[index][j]) for j in range(self.PartiesNumber)]) < self.DistrictSeats[index]]
            self.OverRepresentedDistricts = [index for index in range(self.DistrictsNumber) if sum(
                [int(self.SeatMatrix[index][j]) for j in range(self.PartiesNumber)]) > self.DistrictSeats[index]]

    def UpdateIncrDecrMatrix(self):
        self.IncrDecrMatrix = []
        if len(self.SeatMatrix) == 0:
            raise ValueError("Seat matrix is missing.")
        for i in range(self.DistrictsNumber):
            NewRow = []
            for j in range(self.PartiesNumber):
                if isinstance(self.SeatMatrix[i][j], float):
                    if self.SeatMatrix[i][j] - ma.floor(self.SeatMatrix[i][j]) > 0.02:
                        NewRow += ["I"]
                    else:
                        NewRow += ["D"]
                else:
                    NewRow += [" "]
            self.IncrDecrMatrix += [NewRow]

    def SearchUpdateFactor(self, RowList, ColList=[]):
        if not(isinstance(RowList, list)) or not(all(isinstance(index, int) and index < self.DistrictsNumber for index in RowList)):
            raise TypeError("Row indices must be inferior than " +
                            str(self.DistrictsNumber)+" and must be provided inside a list.")
        if not(isinstance(ColList, list)) or not(all(isinstance(index, int) and index < self.PartiesNumber for index in ColList)):
            raise TypeError("Column indices must be inferior than " +
                            str(self.PartiesNumber)+" and must be provided inside a list.")
        RowListCompl = list(
            set([i for i in range(self.DistrictsNumber)]) - set(RowList))
        ColListCompl = list(
            set([i for i in range(self.PartiesNumber)])-set(ColList))
        Alph = max([0]+[WebsterSignpost(self.SeatMatrix[i][j])/self.WeightMatrix[i][j]
                        for i in RowList for j in ColListCompl if WebsterSignpost(self.SeatMatrix[i][j]) > 0])
        Beta = min([ma.inf]+[WebsterSignpost(self.SeatMatrix[i][j]+1)/self.WeightMatrix[i][j]
                             for i in RowListCompl for j in ColList if self.WeightMatrix[i][j] > 0])
        UpdateFactor = Alph
        if Alph < 1/Beta:
            UpdateFactor = 1/Beta
        return UpdateFactor

    def LatexClipboardData(self, WeightSeat="W", Caption=False, CaptionText="", Label=False, LabelText=""):
        Options = ["W", "S"]
        if not(isinstance(WeightSeat, str)) or WeightSeat not in ["W", "S"]:
            raise ValueError("First argument must be in "+str(Options)+".")
        Output = ""
        Output += "\\begin{table}[h]\n"
        Output += "\\centering\n"
        if WeightSeat == "S":
            Output += "\\begin{tabular}{" + \
                str(LatexTableColumnIndexGenerator(
                    self.PartiesNumber+1))+"c|c|}\n"
        else:
            Output += "\\begin{tabular}{" + \
                str(LatexTableColumnIndexGenerator(self.PartiesNumber+1))+"}\n"
        Output += " \\hline\n"
        HeaderString = "Region/Party"
        for name in self.Parties:
            HeaderString += " & " + name
        if WeightSeat == "S":
            HeaderString += " & Total & Dif."
        HeaderString += " \\\\\n"
        Output += HeaderString
        Output += " \\hline\n"
        for i in range(self.DistrictsNumber):
            RowString = self.Districts[i]
            if WeightSeat == "W":
                for weight in self.WeightMatrix[i]:
                    RowString += " & " + str(weight)
            elif WeightSeat == "S":
                for seat in self.SeatMatrix[i]:
                    if isinstance(seat, float):
                        if seat - ma.floor(seat) > 0.02:
                            RowString += " & " + str(ma.floor(seat))+"+"
                        else:
                            RowString += " & " + str(ma.floor(seat))+"-"
                    else:
                        RowString += " & " + str(seat)
                RowString += " & " + str(self.DistrictSeats[i])
                RowString += " & " + \
                    str(sum([int(seat) for seat in self.SeatMatrix[i]]
                            ) - self.DistrictSeats[i])
            RowString += " \\\\\n"
            Output += RowString
        if WeightSeat == "S":
            Output += " \\hline\nTotal"
            for seat in self.TotalPartySeats:
                Output += " & " + str(seat)
            Output += " & " + str(self.HouseSize) + " & " + " \\\\\n"
        Output += " \\hline\n"
        Output += "\\end{tabular}\n"
        if Caption:
            Output += "\\caption{"+CaptionText+"}\n"
        if Label:
            Output += "\\label{"+LabelText+"}\n"
        Output += "\\end{table}"
        pyclip.copy(Output)

##############################################################
##                         RESULTS                          ##
##############################################################

##############################################################
#                    Convert to Latex
##############################################################


def LatexClipboardDataDoubleColumn(DataMatrix, Headers, Caption=False, CaptionText="", Label=False, LabelText=""):
    if len(DataMatrix[0]) != len(Headers):
        raise ValueError(
            "Las dimensiones de las cabeceras y datos no coinciden.")
    Output = ""
    Output += "\\begin{table}[h]\n"
    Output += "\\centering\n \\footnotesize\n"
    Output += "\\begin{tabular}{" + \
        str(LatexTableColumnIndexGenerator(len(Headers))) + \
        str(LatexTableColumnIndexGenerator(len(Headers)))+"}\n"
    Output += " \\hline\n"
    HeaderString = " "+Headers[0]
    for i in range(1, len(Headers)):
        HeaderString += " & " + Headers[i]
    HeaderString += " & "+Headers[0]
    for i in range(1, len(Headers)):
        HeaderString += " & " + Headers[i]
    HeaderString += " \\\\\n"
    Output += HeaderString
    Output += " \\hline\n"
    for j in range(0, len(DataMatrix), 2):
        RowString = " "
        for i in range(len(DataMatrix[j])):
            RowString += str(DataMatrix[j][i]) + " & "
        if j + 1 != len(DataMatrix):
            for i in range(len(DataMatrix[j+1])-1):
                RowString += str(DataMatrix[j+1][i]) + " & "
            RowString += str(DataMatrix[j+1][-1]) + " \\\\\n"
        else:
            for i in range(len(DataMatrix[j])-1):
                RowString += "- & "
            RowString += "- \\\\\n\\hline"
        Output += RowString
    Output += "\\end{tabular}\n"
    if Caption:
        Output += "\\caption{"+CaptionText+"}\n"
    if Label:
        Output += "\\label{"+LabelText+"}\n"
    Output += "\\end{table}"
    pyclip.copy(Output)

    return True


def LatexClipboardDataOneColumn(DataMatrix, Headers, Caption=False, CaptionText="", Label=False, LabelText=""):
    if len(DataMatrix[0]) != len(Headers):
        raise ValueError(
            "Las dimensiones de las cabeceras y datos no coinciden.")
    Output = ""
    Output += "\\begin{table}[h]\n"
    Output += "\\centering\n \\footnotesize\n"
    Output += "\\begin{tabular}{" + \
        str(LatexTableColumnIndexGenerator(len(Headers)))+"}\n"
    Output += " \\hline\n"
    HeaderString = " "+Headers[0]
    for i in range(1, len(Headers)):
        HeaderString += " & " + Headers[i]
    HeaderString += " \\\\\n"
    Output += HeaderString
    Output += " \\hline\n"
    for j in range(0, len(DataMatrix)):
        RowString = " "
        for i in range(len(DataMatrix[j])-1):
            RowString += str(DataMatrix[j][i]) + " & "
        RowString += str(DataMatrix[j][-1]) + " \\\\\n\\hline\n"
        Output += RowString
    Output += "\\end{tabular}\n"
    if Caption:
        Output += "\\caption{"+CaptionText+"}\n"
    if Label:
        Output += "\\label{"+LabelText+"}\n"
    Output += "\\end{table}"
    pyclip.copy(Output)

    return True


##############################################################
##############################################################
# LOADING DATA
os.chdir(pathlib.Path(__file__).parent.resolve())
# Datos generales 2018
Datos = []
name = 'Datos.csv'
with open(name, 'r') as dat:
    csv_reader = csv.reader(dat, delimiter=',')
    for row in csv_reader:
        Datos += [list(row)]
for i in range(len(Datos)):
    Datos[i][0] = Datos[i][0].replace("\xa0", "")
    Datos[i][1] = int(Datos[i][1])
    Datos[i][2] = float(Datos[i][2])
    Datos[i][3] = int(Datos[i][3])
VectorCabeceras = ["Nombre Partido", "Nº Votos", "Porcentaje", "Escaños"]
# for data visualization
SortedAbbrevNames = ["Avante", "DC", "DEM", "MDB", "PCB", "PC do B", "PDT", "PHS", "NOVO", "PPS", "PP", "PPL", "PRTB", "PRB", "PRP", "PROS", "PSC",
                     "PSD", "PSL", "PSOL", "PSB", "PSTU", "PTB", "PTC", "PV", "PCO", "PMN", "PMB", "PR", "PSDB", "PT", "PATRI", "PODE", "REDE", "SOLID."]
Datos = SortPerIndex(Datos, 0, False)
for i in range(len(Datos)):
    Datos[i][0] = SortedAbbrevNames[i]

# Datos Generales 2014
Datos2 = []
name2 = 'Datos_2014.csv'
with open(name2, 'r') as dat:
    csv_reader = csv.reader(dat, delimiter=',')
    for row in csv_reader:
        Datos2 += [list(row)]
for i in range(len(Datos2)):
    Datos2[i][0] = Datos2[i][0].replace("\xa0", "")
    Datos2[i][1] = int(Datos2[i][1])
    Datos2[i][2] = float(Datos2[i][2])
    Datos2[i][3] = int(Datos2[i][3])
# for data visualization
SortedAbbrevNames2 = ["DEM", "PCB", "PC do B", "PDT", "PEN", "PHS", "PPS", "PP", "PPL", "PRTB", "PRB", "PRP", "PROS", "PSC", "PSDC",
                      "PSD", "PSL", "PSOL", "PSB", "PSTU", "PTB", "PTC", "PTN", "PT do B", "PV", "PCO", "PMN", "PR", "PSDB", "PMDB", "PT", "SD"]
Datos2 = SortPerIndex(Datos2, 0, False)
for i in range(len(Datos2)):
    Datos2[i][0] = SortedAbbrevNames2[i]

# Datos circunscripciones 2018
zipFolder = zp.ZipFile('FU_Folder.zip')
zipFolder.extractall(path=None, members=None, pwd=None)
FU_names = zipFolder.namelist()
FU_names.sort()
DatosEstado = []
for k in range(1, 28):
    DatosCircuns = []
    with open(FU_names[k], 'r') as dat:
        csv_reader = csv.reader(dat, delimiter=',')
        for row in csv_reader:
            DatosCircuns += [list(row)]
    DatosCircuns[1][0] = int(DatosCircuns[1][0])
    for i in range(2, len(DatosCircuns)):
        DatosCircuns[i][0] = DatosCircuns[i][0].replace("\xa0", "")
        if DatosCircuns[i][0][0] == " ":
            DatosCircuns[i][0] = DatosCircuns[i][0][1::]
        DatosCircuns[i][1] = int(DatosCircuns[i][1])
        DatosCircuns[i][2] = float(DatosCircuns[i][2])
        DatosCircuns[i][3] = int(DatosCircuns[i][3])
    DatosEstado += [DatosCircuns]
DatosTest = cp.deepcopy(DatosEstado)
# PREPARING DATA FOR VOTEWEIGHTMATRIX CLASS (2018)

Parties = []
for circuns in DatosTest:
    for i in range(2, len(circuns)):
        if circuns[i][0] not in Parties:
            Parties += [circuns[i][0]]
# opcional, tablas mejores
DatosTestF = []
for i in range(len(DatosTest)):
    Circunsc = []
    Circunsc += [DatosTest[i][0][0], DatosTest[i][1][0]]
    AuxList = []
    VotedParties = []
    for j in range(2, len(DatosTest[i])):
        AuxDict = {"PartyName": DatosTest[i]
                   [j][0], "Votes": DatosTest[i][j][1]}
        VotedParties += [DatosTest[i][j][0]]
        AuxList += [AuxDict]
    for party in Parties:
        if party not in VotedParties:
            AuxDict = {"PartyName": party, "Votes": 0}
            AuxList += [AuxDict]
    Circunsc += [AuxList]
    Circunsc[2] = sorted(Circunsc[2], key=lambda e: (e['PartyName']))
    DatosTestF += [Circunsc]
PartyNames = sorted(Parties)
PartyNames = SortedAbbrevNames
VoteMatrix = [[DatosTestF[i][2][j]["Votes"]
               for j in range(len(PartyNames))] for i in range(len(DatosTestF))]
DistrictNames = [DatosTestF[i][0] for i in range(len(DatosTestF))]
DistrictSizes = [DatosTestF[i][1] for i in range(len(DatosTestF))]

# CREATION OF VOTEWEIGHT OBJET
TestDat = WeightSeatMatrix(VoteMatrix, DistrictNames,
                           PartyNames, DistrictSizes)


# Datos circunscripciones 2014
zipFolder2 = zp.ZipFile('FU_Folder_2014.zip')
zipFolder2.extractall(path=None, members=None, pwd=None)
FU_names2 = zipFolder2.namelist()
FU_names2.sort()
DatosEstado2 = []
for k in range(1, 28):
    DatosCircuns = []
    with open(FU_names2[k], 'r') as dat:
        csv_reader = csv.reader(dat, delimiter=',')
        for row in csv_reader:
            DatosCircuns += [list(row)]
    DatosCircuns[1][0] = int(DatosCircuns[1][0])
    for i in range(2, len(DatosCircuns)):
        DatosCircuns[i][0] = DatosCircuns[i][0].replace("\xa0", "")
        if DatosCircuns[i][0][0] == " ":
            DatosCircuns[i][0] = DatosCircuns[i][0][1::]
        DatosCircuns[i][1] = int(DatosCircuns[i][1])
        DatosCircuns[i][2] = float(DatosCircuns[i][2])
        DatosCircuns[i][3] = int(DatosCircuns[i][3])
    DatosEstado2 += [DatosCircuns]
DatosTest2 = cp.deepcopy(DatosEstado2)
# PREPARING DATA FOR VOTEWEIGHTMATRIX CLASS (2014)

Parties2 = []
for circuns in DatosTest2:
    for i in range(2, len(circuns)):
        if circuns[i][0] not in Parties2:
            Parties2 += [circuns[i][0]]
# opcional, tablas mejores
DatosTestF2 = []
for i in range(len(DatosTest2)):
    Circunsc = []
    Circunsc += [DatosTest2[i][0][0], DatosTest2[i][1][0]]
    AuxList = []
    VotedParties = []
    for j in range(2, len(DatosTest2[i])):
        AuxDict = {"PartyName": DatosTest2[i]
                   [j][0], "Votes": DatosTest2[i][j][1]}
        VotedParties += [DatosTest2[i][j][0]]
        AuxList += [AuxDict]
    for party in Parties2:
        if party not in VotedParties:
            AuxDict = {"PartyName": party, "Votes": 0}
            AuxList += [AuxDict]
    Circunsc += [AuxList]
    Circunsc[2] = sorted(Circunsc[2], key=lambda e: (e['PartyName']))
    DatosTestF2 += [Circunsc]
PartyNames2 = sorted(Parties2)
PartyNames2 = SortedAbbrevNames2
VoteMatrix2 = [[DatosTestF2[i][2][j]["Votes"]
                for j in range(len(PartyNames2))] for i in range(len(DatosTestF2))]
DistrictNames2 = [DatosTestF2[i][0] for i in range(len(DatosTestF2))]
DistrictSizes2 = [DatosTestF2[i][1] for i in range(len(DatosTestF2))]

# CREATION OF VOTEWEIGHT OBJET
TestDat_2014 = WeightSeatMatrix(VoteMatrix2, DistrictNames2,
                                PartyNames2, DistrictSizes2)

#################################################
#               DATA MANIPULATION               #
#################################################
MGNames = [DatosTest[11][i][0]
           for i in range(2, len(DatosTest[11]))]
MGVotes = [DatosTest[11][i][1]
           for i in range(2, len(DatosTest[11]))]
MGSeats = [DatosTest[11][i][3]
           for i in range(2, len(DatosTest[11]))]
Matrix = [MGNames] + [MGVotes]+[MGSeats]
Matrix = [list(i) for i in zip(*Matrix)]
Matrix = SortPerIndex(Matrix, 1)
LatexClipboardDataDoubleColumn(Matrix, ["Partido", "Votos", "Escaños"])
# GENERAL RESULTS 2018, 2014
"""
TieTransfer(TestDat)
General2018 = [TestDat.Parties] + [[Datos[i][1]
                                    for i in range(len(Datos))]] + [[Datos[i][3]
                                                                     for i in range(len(Datos))]] + [TestDat.TotalPartySeats]
General2018 = SortPerIndex([list(i) for i in zip(*General2018)], 1)
# LatexClipboardDataDoubleColumn(
#    General2018, ["Partido", "Votos", "Esc.", "Esc.$^*$"])

TieTransfer(TestDat_2014)
General2014 = [TestDat_2014.Parties] + [[Datos2[i][1]
                                         for i in range(len(Datos2))]] + [[Datos2[i][3]
                                                                           for i in range(len(Datos2))]] + [TestDat.TotalPartySeats]
General2014 = SortPerIndex([list(i) for i in zip(*General2014)], 1)
LatexClipboardDataDoubleColumn(
    General2014, ["Partido", "Votos", "Esc.", "Esc.$^*$"])
"""
# Desproporcionalidad Barreras electorales
# 2018
"""
TotalSeatMatrix = []
n = 7
HeaderList = ["Orig."]
for i in [i*0.065/n for i in range(n+1)]:
    HeaderList += [str(round(100*i, 2))+"\%"]
    VoteMatrixAux = Threshold(VoteMatrix, i)
    TestDatAux = WeightSeatMatrix(
        VoteMatrixAux, DistrictNames, PartyNames, sum(DistrictSizes), DistrictSizes)
    TieTransfer(TestDatAux)
    TotalSeatMatrix += [TestDatAux.TotalPartySeats]
# each list represents the results at a certain threshold (but the first one which is the original outcome)
TotalSeatMatrixPerc = TotalSeatMatrix
for i in range(len(TotalSeatMatrix)):
    TotalSeatMatrixPerc[i] = [round(
        100*TotalSeatMatrix[i][j]/513, 2) for j in range(len(TotalSeatMatrix[i]))]
VotesPercentage = [Datos[i][2] for i in range(len(Datos))]

TotalSeatMatrixPerc = [[round(100*Datos[i][3]/513, 2)
                        for i in range(len(Datos))]] + TotalSeatMatrixPerc
GalList_2018 = [LeastSquareIndex(VotesPercentage, TotalSeatMatrixPerc[i])[
    0] for i in range(len(TotalSeatMatrixPerc))]
JeffList_2018 = [JeffersonIndex(VotesPercentage, TotalSeatMatrixPerc[i], 0)[
    0] for i in range(len(TotalSeatMatrixPerc))]
SaLaList_2018 = [SainteLagueIndex(VotesPercentage, TotalSeatMatrixPerc[i])[
    0] for i in range(len(TotalSeatMatrixPerc))]

# 2014
TotalSeatMatrix2 = []
for i in [i*0.065/n for i in range(n+1)]:
    VoteMatrixAux = Threshold(VoteMatrix2, i)
    TestDatAux = WeightSeatMatrix(
        VoteMatrixAux, DistrictNames2, PartyNames2, sum(DistrictSizes2), DistrictSizes2)
    TieTransfer(TestDatAux)
    TotalSeatMatrix2 += [TestDatAux.TotalPartySeats]
# each list represents the results at a certain threshold (but the first one which is the original outcome)
TotalSeatMatrix2Perc = TotalSeatMatrix2
for i in range(len(TotalSeatMatrix2)):
    TotalSeatMatrix2Perc[i] = [round(
        100*TotalSeatMatrix2[i][j]/513, 2) for j in range(len(TotalSeatMatrix2[i]))]
VotesPercentage2 = [Datos2[i][2] for i in range(len(Datos2))]

TotalSeatMatrix2Perc = [[round(100*Datos2[i][3]/513, 2)
                         for i in range(len(Datos2))]] + TotalSeatMatrix2Perc
GalList_2014 = [LeastSquareIndex(VotesPercentage2, TotalSeatMatrix2Perc[i])[
    0] for i in range(len(TotalSeatMatrix2Perc))]
JeffList_2014 = [JeffersonIndex(VotesPercentage2, TotalSeatMatrix2Perc[i], 0)[
    0] for i in range(len(TotalSeatMatrix2Perc))]
SaLaList_2014 = [SainteLagueIndex(VotesPercentage2, TotalSeatMatrix2Perc[i])[
    0] for i in range(len(TotalSeatMatrix2Perc))]


TotalIndexMatrix = [["BRA-2018: I$_G$"]+GalList_2018, ["BRA-2018: I$_J$"]+JeffList_2018,
                    ["BRA-2018: I$_{S-L}$"]+SaLaList_2018, ["BRA-2014: I$_G$"]+GalList_2014, ["BRA-2014: I$_J$"]+JeffList_2014, ["BRA-2014: I$_{S-L}$"]+SaLaList_2014]
HeaderList = ["Índices/Barreras"] + HeaderList
# print(TotalIndexMatrix[0])
#LatexClipboardDataOneColumn(TotalIndexMatrix, HeaderList)
"""
# DESPROPROCIONALIDAD UNIDADES FEDERATIVAS
"""
TieTransfer(TestDat)
FuNames = [DatosTest[i][0][0] for i in range(len(DatosTest))]
IndexMatrix = []
for i in range(len(DatosTest)):
    AuxRow = []
    OriginalVotes = [DatosTest[i][j][1] for j in range(2, len(DatosTest[i]))]
    PercentageOriginalVotes = [
        round(100*element/sum(OriginalVotes), 2) for element in OriginalVotes]
    OriginalSeats = [DatosTest[i][j][3] for j in range(2, len(DatosTest[i]))]
    PercentageOriginalSeats = [
        round(100*element/sum(OriginalSeats), 2) for element in OriginalSeats]
    OrigGal = LeastSquareIndex(
        PercentageOriginalVotes, PercentageOriginalSeats)[0]
    OrigJef = JeffersonIndex(PercentageOriginalVotes,
                             PercentageOriginalSeats)[0]
    OrigSaLa = SainteLagueIndex(
        PercentageOriginalVotes, PercentageOriginalSeats)[0]
    NewVotes = TestDat.VoteMatrix[i]
    PercentageNewVotes = [round(100*votes/sum(NewVotes), 2)
                          for votes in NewVotes]
    NewSeats = TestDat.SeatMatrix[i]
    PercentageNewSeats = [round(100*seats/sum(NewSeats), 2)
                          for seats in NewSeats]
    NewGal = LeastSquareIndex(
        PercentageNewVotes, PercentageNewSeats)[0]
    NewJef = JeffersonIndex(PercentageNewVotes,
                            PercentageNewSeats)[0]
    NewSaLa = SainteLagueIndex(
        PercentageNewVotes, PercentageNewSeats)[0]
    AuxRow += [OrigGal, NewGal, OrigJef, NewJef, OrigSaLa, NewSaLa]

    IndexMatrix += [[FuNames[i]] + AuxRow]

LatexClipboardDataDoubleColumn(IndexMatrix, [
    "FU", "I$_G$", "I$_G^*$", "I$_J$", "I$_J^*$", "I$_{S-L}^*$", "I$_{S-L}^*$"])
"""
# REPRESENTACION DE ESCAÑOS POR UNIDAD FEDERATIVA
"""
TieTransfer(TestDat_2014)
FirstMatrix = [TestDat_2014.TotalPartySeats]
Stop1 = int(len(TestDat_2014.SeatMatrix)/3)
Stop2 = 2*int(len(TestDat_2014.SeatMatrix)/3)
for i in range(Stop1):
    FirstMatrix += [TestDat_2014.SeatMatrix[i]]
for i in range(len(FirstMatrix)):
    for j in range(len(FirstMatrix[i])):
        FirstMatrix[i][j] = int(FirstMatrix[i][j])
FirstMatrix = [TestDat_2014.Parties] + FirstMatrix
FirstMatrix = [list(i) for i in zip(*FirstMatrix)]
FirstMatrix = SortPerIndex(FirstMatrix, 1)
for i in range(len(FirstMatrix)):
    FirstMatrix[i].pop(1)
FirstHeader = ["BR" for element in FirstMatrix[0]]
#LatexClipboardDataOneColumn(FirstMatrix, FirstHeader)

SecondMatrix = [TestDat_2014.TotalPartySeats]
for i in range(Stop1, Stop2):
    SecondMatrix += [TestDat_2014.SeatMatrix[i]]
for i in range(len(SecondMatrix)):
    for j in range(len(SecondMatrix[i])):
        SecondMatrix[i][j] = int(SecondMatrix[i][j])
SecondMatrix = [TestDat_2014.Parties] + SecondMatrix
SecondMatrix = [list(i) for i in zip(*SecondMatrix)]
SecondMatrix = SortPerIndex(SecondMatrix, 1)
for i in range(len(SecondMatrix)):
    SecondMatrix[i].pop(1)
SecondHeader = ["BR" for element in SecondMatrix[0]]
#LatexClipboardDataOneColumn(SecondMatrix, SecondHeader)

ThirdMatrix = [TestDat_2014.TotalPartySeats]
for i in range(Stop2, len(TestDat_2014.SeatMatrix)):
    ThirdMatrix += [TestDat_2014.SeatMatrix[i]]
for i in range(len(ThirdMatrix)):
    for j in range(len(ThirdMatrix[i])):
        ThirdMatrix[i][j] = int(ThirdMatrix[i][j])
ThirdMatrix = [TestDat_2014.Parties] + ThirdMatrix
ThirdMatrix = [list(i) for i in zip(*ThirdMatrix)]
ThirdMatrix = SortPerIndex(ThirdMatrix, 1)
for i in range(len(ThirdMatrix)):
    ThirdMatrix[i].pop(1)
ThirdHeader = ["BR" for element in ThirdMatrix[0]]
LatexClipboardDataOneColumn(ThirdMatrix, ThirdHeader)
"""
# indices
"""
DatosOriginales = SortPerIndex(Datos2, 0, False)
VotosTotales = TestDat_2014.TotalPartyVotes
votos2 = [round(100*votos/sum(VotosTotales), 2) for votos in VotosTotales]
votos = [DatosOriginales[i][2] for i in range(len(DatosOriginales))]
EscanosTotales = TestDat_2014.TotalPartySeats
EscanosOriginales = [round(100*DatosOriginales[i][3]/sum(EscanosTotales), 2)
                     for i in range(len(DatosOriginales))]
escanos = [round(100*EscanosTotales[i]/sum(EscanosTotales), 2)
           for i in range(len(EscanosTotales))]
print(escanos)
# print(SainteLagueIndex(votos, escanos, 6))

# ShowMatrix = [PartyNames + ["Índice"]] + [votos+[-1]] + [EscanosOriginales +
#                                                         ["-"]] + [escanos+["-"]]
ShowMatrix = []
Lh1 = LoosemoreHanbyIndex(votos, EscanosOriginales)
Lh2 = LoosemoreHanbyIndex(votos2, escanos)
ShowMatrix += [[Lh1[0], Lh2[0]]]
LS1 = LeastSquareIndex(votos, EscanosOriginales)
LS2 = LeastSquareIndex(votos2, escanos)
ShowMatrix += [[LS1[0], LS2[0]]]
SL1 = SainteLagueIndex(votos, EscanosOriginales)
SL2 = SainteLagueIndex(votos2, escanos)
ShowMatrix += [[SL1[0], SL2[0]]]
J1 = JeffersonIndex(votos, EscanosOriginales)
J2 = JeffersonIndex(votos2, escanos)
ShowMatrix += [[J1[0], J2[0]]]
#Indices = [list(i) for i in zip(*ShowMatrix)]
#Indices = SortPerIndex(Indices, 1)
Cabecera = ["Partido", "\% Votos", "\% Esc.", "\% Esc.2",
            "Gal.", "Gal.2", "S-L", "S-L.2", "D'H", "D'H.2"]
print(ShowMatrix)


def LatexClipboardData(DataMatrix, Headers, Caption=False, CaptionText="", Label=False, LabelText=""):
    if len(DataMatrix[0]) != len(Headers):
        raise ValueError(
            "Las dimensiones de las cabeceras y datos no coinciden.")
    Output = ""
    Output += "\\begin{table}[h]\n"
    Output += "\\centering\n"
    Output += "\\begin{tabular}{" + \
        str(LatexTableColumnIndexGenerator(len(Headers)))+"}\n"
    Output += " \\hline\n"
    HeaderString = " "+Headers[0]
    for i in range(1, len(Headers)):
        HeaderString += " & " + Headers[i]
    HeaderString += " \\\\\n"
    Output += HeaderString
    Output += " \\hline\n"
    for row in DataMatrix:
        RowString = " "
        for i in range(len(row)-1):
            RowString += str(row[i]) + " & "
        RowString += str(row[-1]) + " \\\\\n"
        Output += RowString
    Output += "\\end{tabular}\n"
    if Caption:
        Output += "\\caption{"+CaptionText+"}\n"
    if Label:
        Output += "\\label{"+LabelText+"}\n"
    Output += "\\end{table}"
    pyclip.copy(Output)

    return True


#LatexClipboardData(Indices, Cabecera)
"""
# Objetivo mostrar barreras en tabla y grafica
"""
PartyNames = ["Avante", "DC", "DEM", "MDB", "PCB", "PC do B", "PDT", "PHS", "NOVO", "PPS", "PP", "PPL", "PRTB", "PRB", "PRP", "PROS", "PSC",
              "PSD", "PSL", "PSOL", "PSB", "PSTU", "PTB", "PTC", "PV", "PCO", "PMN", "PMB", "PR", "PSDB", "PT", "PATRI", "PODE", "REDE", "SOLID."]
TotalVoteMatrix = []
n = 9
TopeIni = 10
TopePartidosQuePlot = 31
for i in [i*0.065/n for i in range(n+1)]:
    VoteMatrixAux = Threshold(VoteMatrix, i)
    TestDatAux = VoteWeightMatrix(
        VoteMatrixAux, DistrictNames, PartyNames, sum(DistrictSizes), DistrictSizes)
    TieTransfer(TestDatAux)
    TotalVoteMatrix += [TestDatAux.TotalPartySeats]
DefinitionInterval = [round(100*i*0.065/n, 1) for i in range(n+1)]
TotalVoteMatrixAux = [list(i) for i in zip(*TotalVoteMatrix)]
colors = [(0.95, 8/256, 32/256), (203/256, 64/256, 176/256), (112/256, 44/256, 151/256),
          (67/256, 98/256, 172/256), (78/256, 124/256, 178/256), (103/256, 195/256, 218/256), (82/256, 208/256, 145/256)]
for i in range(TopeIni, TopePartidosQuePlot):
    plt.plot(DefinitionInterval,
             TotalVoteMatrixAux[i], color=colors[i % 7])
plt.legend(PartyNames[TopeIni:TopePartidosQuePlot])
plt.ylabel("Escaños")
plt.xlabel("% barrera")
plt.show()
TotalTotal = [PartyNames] + TotalVoteMatrix
TotalTotal = [list(i) for i in zip(*TotalTotal)]
TotalTotal = SortPerIndex(TotalTotal, 1)
Cabeceras = ["Partido/Barrera"] + [str(i)+"\%" for i in DefinitionInterval]
print(len(TotalTotal))
print(len(Cabeceras))

LatexClipboardData(TotalTotal, Cabeceras)
"""

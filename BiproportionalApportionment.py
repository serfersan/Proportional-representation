
####################################################################
##         BIPROPORTIONAL ALGORITHM (D'HONDT + WEBSTER)         ##
####################################################################

# Libraries
import os as os
import csv as csv
from tokenize import Number
from typing import List, Tuple
import zipfile as zp
import math as ma
import pyperclip as pyclip
import copy as cp

import pathlib

############################################
#            AUXILIARY FUNCTIONS           #
############################################


def SortPerIndex(OriginalData: List[list], Index=0, Reverse=True) -> List[List]:
    """
    Sorts list of lists given index.

    >>> SortPerIndex([list1, list2, list3, ...],int)
    """
    if not isinstance(OriginalData, list):
        raise TypeError("First argument must be a list of lists.")
    if not all(isinstance(element, list) for element in OriginalData):
        raise TypeError("First argument must be a list of lists.")
    if not isinstance(Index, int):
        raise TypeError("Second argument must be an integer.")
    if not all(len(element) > Index for element in OriginalData):
        raise ValueError(
            "Index provided exceeds length of at least one of the lists given.")
    Data = cp.deepcopy(OriginalData)
    try:
        Data.sort(key=lambda x: x[Index])
        if Reverse:
            Data.reverse()
        return Data
    except:
        raise TypeError(
            "Values could not be ordered.")


def Threshold(OriginalData: list, Percentage: float) -> List[List[int]]:
    """
    Applies an electoral threshold to a matrix of votes in each row. Each element of the matrix\n
    that does not verify the percentage conditions goes to 0.
    >>> Threshold([[V11,V12,V13...],[V21,V22,V23,...],...,[Vn1,Vn2,Vn3,...]], p)
    """
    if not isinstance(OriginalData, list):
        raise TypeError("First argument must be a list of lists.")
    if not all(isinstance(element, list) for element in OriginalData):
        raise TypeError("First argument must be a list of lists.")
    if not all(all(isinstance(votes, int) and votes >= 0 for votes in votelist) for votelist in OriginalData):
        raise ValueError(
            "First argument must be a list of lists composed of nonnegative integers.")
    if not isinstance(Percentage, float):
        raise TypeError("Second argument must be a number in [0,1].")
    if Percentage < 0 or Percentage > 1:
        raise ValueError("Second argument must be in [0,1].")
    ThresholdData = cp.deepcopy(OriginalData)
    RowNumber = len(OriginalData)
    ColNumber = len(OriginalData[0])
    TotalVotes = [sum([OriginalData[i][j] for j in range(ColNumber)])
                  for i in range(RowNumber)]
    for i in range(RowNumber):
        for j in range(ColNumber):
            if OriginalData[i][j]/TotalVotes[i] < Percentage:
                ThresholdData[i][j] = 0
    return ThresholdData


def RoundPosMatrix(Matrix: List[List[float]], Decimal=2) -> List[List[float]]:
    """
    Returns the list of lists given with each element rounded to the decimal place indicated in the second argument.
    """
    if not(isinstance(Matrix, list)) or not(all(isinstance(element, list) for element in Matrix)) or not(all(all(type(number) in [float, int] and number >= 0 for number in element) for element in Matrix)) or not(all(len(element) == len(Matrix[0]) for element in Matrix)):
        raise TypeError(
            "First argument must be a list of lists of equal length composed of nonnegative numbers.")
    if not(isinstance(Decimal, int) and Decimal >= 0):
        raise TypeError("Second argument must be a nonnegative integer.")
    return [[round(Matrix[i][j], Decimal) for j in range(len(Matrix[i]))] for i in range(len(Matrix))]

############################################
#       DESPROPORTIONATE INDICES           #
############################################


def LoosemoreHanbyIndex(Votes: list, Seats: list, ReturnTerms=True) -> Tuple[float, List[float]]:
    """
    Calculates the Loosemore-Hanby index given the seat and vote percentage vectors.\n
    If ReturnTerms = True it also returns the terms involved in the calculation of\n
    the index, |vi-si|.
    >>> LoosemoreHanbyIndex([v1,v2,v3,...],[s1,s2,s3,...]) -> (float,list)
    """
    if not (isinstance(Votes, list) and all(isinstance(vote, float) or isinstance(vote, int) for vote in Votes)):
        raise TypeError(
            "First argument must be a list of float numbers in [0,100]")
    if not all(vote >= 0 and vote <= 100 for vote in Votes):
        raise ValueError(
            "First argument must be a list of float numbers in [0,100]")
    if not (isinstance(Seats, list) and all(isinstance(seat, float) or isinstance(seat, int) for seat in Seats)):
        raise TypeError(
            "First argument must be a list of float numbers in [0,100]")
    if not all(seat >= 0 and seat <= 100 for seat in Seats):
        raise ValueError(
            "First argument must be a list of float numbers in [0,100]")
    if not len(Seats) == len(Votes):
        return ValueError("Both arguments must be lists of equal length.")
    if not isinstance(ReturnTerms, bool):
        raise TypeError("Third argument must be boolean.")
    DiffList = []
    for (vote, seat) in zip(Votes, Seats):
        DiffList += [round(abs(vote - seat), 2)]
    Index = round(sum(DiffList)/2, 2)
    if ReturnTerms:
        return (Index, DiffList)
    else:
        return Index


def RaeIndex(Votes: list, Seats: list, ReturnTerms=True) -> Tuple[float, List[float]]:
    """
    Calculates the Rae index given the seat and vote percentage vectors.\n
    If ReturnTerms = True it also returns the terms involved in the calculation of\n
    the index, |vi-si|.
    >>> RaeIndex([v1,v2,v3,...],[s1,s2,s3,...]) -> (float,list)
    """
    if not (isinstance(Votes, list) and all(isinstance(vote, float) or isinstance(vote, int) for vote in Votes)):
        raise TypeError(
            "First argument must be a list of float numbers in [0,100]")
    if not all(vote >= 0 and vote <= 100 for vote in Votes):
        raise ValueError(
            "First argument must be a list of float numbers in [0,100]")
    if not (isinstance(Seats, list) and all(isinstance(seat, float) or isinstance(seat, int) for seat in Seats)):
        raise TypeError(
            "First argument must be a list of float numbers in [0,100]")
    if not all(seat >= 0 and seat <= 100 for seat in Seats):
        raise ValueError(
            "First argument must be a list of float numbers in [0,100]")
    if not len(Seats) == len(Votes):
        return ValueError("Both arguments must be lists of equal length.")
    if not isinstance(ReturnTerms, bool):
        raise TypeError("Third argument must be boolean.")
    DiffList = []
    cont = 0
    for (vote, seat) in zip(Votes, Seats):
        if vote > 0.5:
            DiffList += [round(abs(vote - seat), 2)]
            cont += 1
        else:
            DiffList += [0]
    try:
        Index = round(sum(DiffList)/cont, 2)
        if ReturnTerms:
            return (Index, DiffList)
        else:
            return Index
    except:
        raise ValueError(
            "All values of the vote vector are inferior than 0.5.")


def LeastSquareIndex(Votes: list, Seats: list, Rounding=2, ReturnTerms=True) -> Tuple[float, List[float]]:
    """
    Calculates the Gallagher index given the seat and vote percentage vectors.\n
    If ReturnTerms = True it also returns the terms involved in the calculation of\n
    the index, (vi-si)^2.
    >>> LeastSquareIndex([v1,v2,v3,...],[s1,s2,s3,...], n) -> (float,list)
    """
    if not (isinstance(Votes, list) and all(isinstance(vote, float) or isinstance(vote, int) for vote in Votes)):
        raise TypeError(
            "First argument must be a list of float numbers in [0,100]")
    if not all(vote >= 0 and vote <= 100 for vote in Votes):
        raise ValueError(
            "First argument must be a list of float numbers in [0,100]")
    if not (isinstance(Seats, list) and all(isinstance(seat, float) or isinstance(seat, int) for seat in Seats)):
        raise TypeError(
            "First argument must be a list of float numbers in [0,100]")
    if not all(seat >= 0 and seat <= 100 for seat in Seats):
        raise ValueError(
            "First argument must be a list of float numbers in [0,100]")
    if not len(Seats) == len(Votes):
        return ValueError("Both arguments must be lists of equal length.")
    if not (isinstance(Rounding, int) and Rounding >= 0):
        raise TypeError("Third argument must be a nonnegative integer.")
    if not isinstance(ReturnTerms, bool):
        raise TypeError("Fourth argument must be boolean.")
    DiffList = []
    for (vote, seat) in zip(Votes, Seats):
        DiffList += [round((vote - seat)**2, Rounding)]
    Index = round((sum(DiffList)/2)**(1/2), Rounding)
    if ReturnTerms:
        return (Index, DiffList)
    else:
        return Index


def SainteLagueIndex(Votes: list, Seats: list, Rounding=2, ReturnTerms=True) -> Tuple[float, List[float]]:
    """
    Calculates the Rae index given the seat and vote percentage vectors.\n
    If ReturnTerms = True it also returns the terms involved in the calculation of\n
    the index, (vi - si)^2/vi.
    >>> SainteLagueIndex([v1,v2,v3,...],[s1,s2,s3,...]) -> (float,list)
    """
    if not (isinstance(Votes, list) and all(isinstance(vote, float) or isinstance(vote, int) for vote in Votes)):
        raise TypeError(
            "First argument must be a list of float numbers in [0,100]")
    if not all(vote >= 0 and vote <= 100 for vote in Votes):
        raise ValueError(
            "First argument must be a list of float numbers in [0,100]")
    if not (isinstance(Seats, list) and all(isinstance(seat, float) or isinstance(seat, int) for seat in Seats)):
        raise TypeError(
            "First argument must be a list of float numbers in [0,100]")
    if not all(seat >= 0 and seat <= 100 for seat in Seats):
        raise ValueError(
            "First argument must be a list of float numbers in [0,100]")
    if not len(Seats) == len(Votes):
        return ValueError("Both arguments must be lists of equal length.")
    if not (isinstance(Rounding, int) and Rounding >= 0):
        raise TypeError("Third argument must be a nonnegative integer.")
    if not isinstance(ReturnTerms, bool):
        raise TypeError("Fourth argument must be boolean.")
    DiffList = []
    for (vote, seat) in zip(Votes, Seats):
        if seat > 0 and vote > 0:
            DiffList += [round((vote - seat)**2/vote, Rounding)]
        else:
            DiffList += [0]
    Index = round(sum(DiffList), Rounding)
    if ReturnTerms:
        return (Index, DiffList)
    else:
        return Index


def JeffersonIndex(Votes: list, Seats: list, Rounding=2, Percentage=0.0, ReturnTerms=True) -> Tuple[float, List[float]]:
    """
    Calculates the Rae index given the seat and vote percentage vectors.\n
    If ReturnTerms = True it also returns the terms involved in the calculation of\n
    the index, si/vi.
    >>> JeffersonIndex([v1,v2,v3,...],[s1,s2,s3,...]) -> (float,list)
    """
    if not (isinstance(Votes, list) and all(isinstance(vote, float) or isinstance(vote, int) for vote in Votes)):
        raise TypeError(
            "First argument must be a list of float numbers in [0,100]")
    if not all(vote >= 0 and vote <= 100 for vote in Votes):
        raise ValueError(
            "First argument must be a list of float numbers in [0,100]")
    if not (isinstance(Seats, list) and all(isinstance(seat, float) or isinstance(seat, int) for seat in Seats)):
        raise TypeError(
            "First argument must be a list of float numbers in [0,100]")
    if not all(seat >= 0 and seat <= 100 for seat in Seats):
        raise ValueError(
            "First argument must be a list of float numbers in [0,100]")
    if not len(Seats) == len(Votes):
        return ValueError("Both arguments must be lists of equal length.")
    if not (isinstance(Rounding, int) and Rounding >= 0):
        raise TypeError("Third argument must be a nonnegative integer.")
    if not ((isinstance(Percentage, float) or isinstance(Percentage, int)) and Percentage >= 0 and Percentage <= 100):
        raise TypeError("Fourth argument must be a number in [0,100].")
    if not isinstance(ReturnTerms, bool):
        raise TypeError("Fifth argument must be boolean.")
    DiffList = []
    for (vote, seat) in zip(Votes, Seats):
        if seat > Percentage and vote > 0:
            DiffList += [round(seat/vote, Rounding)]
        else:
            DiffList += [0]
    Index = max(DiffList)
    if ReturnTerms:
        return (Index, DiffList)
    else:
        return Index

############################################
#         LATEX TABULAR FUNCTIONS          #
############################################


def LatexTableColumnIndexGenerator(ColNumber: int) -> str:
    """
    Auxiliary function for latex table creation functions. Returns a str containing\n
    the columns alignments for latex tabular environment.
    """
    if not isinstance(ColNumber, int):
        raise TypeError("Argument must be a positiv integer.")
    if not ColNumber > 0:
        raise ValueError("Argument must be positive.")
    Output = ""
    aux = ['|c|']+['r' for i in range(ColNumber-1)]+['|']
    return Output.join(aux)


def LatexClipboardData(DataMatrix: List[list], Headers: list, Caption=False, CaptionText="", Label=False, LabelText="") -> pyclip:
    """
    Creates a latex table given de data in first argument and headers in the second argument. Latex code is copied to clipboard.
    """
    if not isinstance(DataMatrix, list):
        raise TypeError("First argument must be a list of lists.")
    if not all(isinstance(element, list) for element in DataMatrix):
        raise TypeError("First argument must be a list of lists.")
    if not isinstance(Headers, list):
        raise ValueError("Second argument must be a list.")
    if len(DataMatrix[0]) != len(Headers):
        raise ValueError(
            "Header and data dimensions are incompatible.")
    if not isinstance(Caption, bool):
        raise TypeError("Third argument must be boolean.")
    if not isinstance(CaptionText, str):
        raise TypeError("Fourth argument must be a string.")
    if not isinstance(Label, bool):
        raise TypeError("Fifth argument must be boolean.")
    if not isinstance(LabelText, str):
        raise TypeError("Sixth argument must be a string.")
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

    return None


def LatexClipboardDataDoubleColumn(DataMatrix: List[list], Headers: List[str], Caption=False, CaptionText="", Label=False, LabelText="") -> pyclip:
    """
    Creates a latex table given de data in first argument and headers in the second argument, the\n
    table separates in two columns the data. Latex code is copied to clipboard.
    """
    if not isinstance(DataMatrix, list):
        raise TypeError("First argument must be a list of lists.")
    if not all(isinstance(element, list) for element in DataMatrix):
        raise TypeError("First argument must be a list of lists.")
    if not isinstance(Headers, list):
        raise ValueError("Second argument must be a list.")
    if len(DataMatrix[0]) != len(Headers):
        raise ValueError(
            "Header and data dimensions are incompatible.")
    if not isinstance(Caption, bool):
        raise TypeError("Third argument must be boolean.")
    if not isinstance(CaptionText, str):
        raise TypeError("Fourth argument must be a string.")
    if not isinstance(Label, bool):
        raise TypeError("Fifth argument must be boolean.")
    if not isinstance(LabelText, str):
        raise TypeError("Sixth argument must be a string.")
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

    return None

############################################
#             DIVISOR METHODS              #
############################################


def WebsterSignpost(Num: Number) -> Number:
    """
    Returns the signposts associated with the standard rule of rounding.
    """
    if not(type(Num) in [int, float]):
        raise TypeError("Argument must be a nonnegative number.")
    if Num < 0:
        raise ValueError("Argument must be nonnegative.")
    if Num == 0:
        return 0
    else:
        return round(Num-0.5, 1)


def WebsterSeatAllocation(Num: Number, Style="M") -> Number:
    """
    Assigns seat following the Sainte-Lague method of divisors. In case of ties\n
    parameter "Style" indicates which of the two possible values returns:\n
    - "M": It returns the higher value.
    - "m": It returns the lower value.
    - "I": It returns the lower value, but indicates with a +0.1 that there was a tie.
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


def Jefferson(VoteList: list, HouseSize: int) -> List[int]:
    """
    Returns a list containing the allocations of seats given a list of votes\n
    and the House size using d'Hondt's method of seat apportionment.
    """
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


def Webster(VoteList: list, HouseSize: int, ReturnWeights=True) -> list:
    """
    Returns a list containing the allocations of seats given a list of votes\n
    and the House size using Sainte-Lague's method of seat apportionment.
    """
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
    if all(votes == 0 for votes in VoteList):
        if ReturnWeights:
            return ([0 for votes in VoteList], [0 for votes in VoteList])
        else:
            return [0 for votes in VoteList]
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

        if ReturnWeights:
            return (SeatList, WeightList)
        else:
            return SeatList
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

        if ReturnWeights:
            return (SeatList, WeightList)
        else:
            return SeatList
    else:
        SeatList = [WebsterSeatAllocation(weight, "I")
                    for weight in WeightList]
        while sum([ma.floor(seat) for seat in SeatList]) < HouseSize:
            SeatList[[index for index in range(len(SeatList)) if isinstance(SeatList[index], float) and round(SeatList[index]-ma.floor(
                SeatList[index]), 2) == 0.1][0]] = round(SeatList[[index for index in range(len(SeatList)) if isinstance(SeatList[index], float) and round(SeatList[index]-ma.floor(
                    SeatList[index]), 2) == 0.1][0]]+0.91, 2)
        SeatList = [round(seat, 2) for seat in SeatList]
        WeightList = [round(weight, 2) for weight in WeightList]

        if ReturnWeights:
            return [SeatList, WeightList]
        else:
            return SeatList


def DirectWebster(WeightList: list, HouseSize: int) -> list:
    """
    Returns a list containing the allocations of seats given a list of votes\n
    and the House size using Sainte-Lague's method of seat apportionment without\n
    altering weights.
    """
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

############################################
#  AUXILIARY FUNCTIONS (BIPROP. ALGORIHTM) #
############################################


def ColumnFinder(List: list, ForbiddenColumns: list) -> list:
    """
    Looks for indices with decrement options within the given list. Returns "E" if there are none.
    """
    if not isinstance(List, list):
        raise TypeError(
            "First argument must be a list of strings.")
    if not all(isinstance(value, str) for value in List):
        raise TypeError(
            "First argument must be a list of strings.")
    if not isinstance(ForbiddenColumns, list):
        raise TypeError(
            "Second argument must be a list of nonnegative integers.")
    if not all(isinstance(value, int) or isinstance(value, float) for value in ForbiddenColumns):
        raise TypeError(
            "Second argument must be a list of nonnegative integers.")
    if any(value >= len(List) for value in ForbiddenColumns):
        raise ValueError(
            "Second argument must contain nonnegative integers with values inferior than length of first argument")
    ColumnOptions = [index for index in range(len(List)) if not(
        index in ForbiddenColumns) and List[index] == "D"]
    if len(ColumnOptions) > 0:
        return ColumnOptions
    else:
        return "E"


def RowFinder(List: list, ForbiddenRows: List[int]) -> list:
    """
    Looks for indices with increment options within the given list. Returns "E" if there are none.
    """
    if not isinstance(List, list):
        raise TypeError(
            "First argument must be a list of strings.")
    if not all(isinstance(value, str) for value in List):
        raise TypeError(
            "First argument must be a list of strings.")
    if not isinstance(ForbiddenRows, list):
        raise TypeError(
            "Second argument must be a list of nonnegative integers.")
    if not all(isinstance(value, int) or isinstance(value, float) for value in ForbiddenRows):
        raise TypeError(
            "Second argument must be a list of nonnegative integers.")
    if any(value >= len(List) for value in ForbiddenRows):
        raise ValueError(
            "Second argument must contain nonnegative integers with values inferior than length of first argument")
    RowOptions = [index for index in range(len(List)) if not(
        index in ForbiddenRows) and List[index] == "I"]
    if len(RowOptions) > 0:
        return RowOptions
    else:
        return "E"


def PathFinder(IncrDecrMatrix: List[list], StartRows: list, EndRows: list, Path=[], StartColumns=[], RowColumn="C") -> List[List[int]]:
    """
    Returns a list of pairs of indices starting in a decrement option in a row from the second argument, it ends\n
    in a increment option in a row from the third argument.
    """
    if not isinstance(IncrDecrMatrix, list):
        raise TypeError("First argument must be a list of lists.")
    if not all(isinstance(element, list) for element in IncrDecrMatrix):
        raise TypeError("First argument must be a list of lists.")
    if not isinstance(StartRows, list):
        raise TypeError(
            "Second argument must be a list containing nonnegative integers.")
    if not all(isinstance(value, int) and value >= 0 for value in StartRows):
        raise TypeError(
            "Second argument must be a list containing nonnegative integers.")
    if not isinstance(EndRows, list):
        raise TypeError(
            "Third argument must be a list containing nonnegative integers.")
    if not all(value < len(IncrDecrMatrix) for value in StartRows):
        raise ValueError(
            "Third argument values must be inferior than the length of the first argument.")
    if not all(value < len(IncrDecrMatrix) for value in EndRows):
        raise ValueError(
            "Fourth argument values must be inferior than the length of the first argument.")
    if not all(isinstance(value, int) and value >= 0 for value in EndRows):
        raise TypeError(
            "Third argument must be a list containing nonnegative integers.")
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


def UpdateLabeled(IncDecMatrix: List[list], LabeledRows: List[int], LabeledColumns=[]) -> List[list]:
    """
    Returns updated labeled rows and columns given a list of lists indicating\n
    increment and decrement options.
    """
    if not isinstance(IncDecMatrix, list):
        raise TypeError("First argument must be a list of lists.")
    if not all(isinstance(element, list) for element in IncDecMatrix):
        raise TypeError("First argument must be a list of lists.")
    if not isinstance(LabeledRows, list):
        raise TypeError(
            "Second argument must be a list containing nonnegative integers.")
    if not all(isinstance(value, int) and value >= 0 for value in LabeledRows):
        raise TypeError(
            "Second argument must be a list containing nonnegative integers.")
    if not isinstance(LabeledColumns, list):
        raise TypeError(
            "Third argument must be a list containing nonnegative integers.")
    if not all(value < len(IncDecMatrix) for value in LabeledRows):
        raise ValueError(
            "Third argument values must be inferior than the length of the first argument.")
    if not all(value < len(IncDecMatrix[0]) for value in LabeledColumns):
        raise TypeError(
            "Third argument must be a list containing nonnegative integers.")
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

###################################################
#             WEIGHTSEATMATRIX CLASS              #
###################################################


class WeightSeatMatrix:
    """
    WeightSeatMatrix class designed for proportional seat apportionment and\
    electoral data manipulation.

    >>> WeightSeatMatrix([[123,231],[421,131]],["District1","District2"],["Party1","Party2"],[size1,size2])

    Parameters
    -----------
    - VoteMatrix: List of lists containing votes from ith electoral district, jth party.
    - DistrictNames: Names of each electoral district.
    - PartyNames: Names of each party.
    - DistrictSizes: Number of seats of each electoral district.

    Atributes
    ----------
    - VoteMatrix: Equals to VoteMatrix parameter given.
    - Housesize: Sum of the seats of each electoral district.
    - TotalPartyVotes: List that consists of the total votes obtained by each party.
    - TotalPartySeats: List of total seats obtained by each party (allocated using d'Hont).
    - DistrictSeats: Equals to DistrictSizes parameter given.
    - TotalVotes: Sum of the total votes of each party.
    - Districts: 
    - DistrictsNumber: Number of districts given.
    - Parties: 
    - PartiesNumber: Number of parties given.
    - WeightMatrix: List of lists with entries each element of the VoteMatrix list of lists\n
     multiplied by its Hare quota.
    - SeatMatrix: List of lists with entries the seat allocation results.
    - IncrDecrMatrix: List of lists with cells "D", "I", " " if there is a decrement option,\n
     increment option or no tie respectively in  its respectives indices within the SeatMatrix.
    - OverRepresentedDistricts: List of indices of overrepresented rows.
    - UnderRepresentedDistricts: List of indices of underrepresented rows.

    Methods
    ----------
    - RowColumnDivisors: Multiplies row and column divisor by given factors.
    - UpdateSeatMatrix: Updates SeatMatrix atribute by applying Webster to the WeightMatrix atribute.
    - UpdateIncrDecrMatrix: Updates IncrDecrMatrix depending on the values of SeatMatrix atribute.
    - SearchUpdateFactor: Returns a value that creates new tie within SeatMatrix atribute.
    - LatexClipboardData: Copies to clipboard WeightMatrix or SeatMatrix tabulated in latex.
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
        Updates WeightMatrix by multiplying row and column divisors by given factors.
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
        """
        Updates SeatMatrix atribute using Sainte-Lague on the WeightMatrix atribute.
        """
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
        """
        Updates IncrDecrMatrix atribute considering the values of SeatMatrix atribute.
        """
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

    def SearchUpdateFactor(self, RowList: List[int], ColList=[]) -> float:
        """
        Returns the closest value to 1 which creates a new tie within the SeatMAtrix atribute.
        """
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
        """
        Creates latex code for importing WeightMatrix or SeatMatrix atributes to tabular environment.\n
        Parameter specifies if the table would contain votes/weights or seats, which would also show\n
        overrepresentation and underrepresentation.
        """
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

############################################
#         BIPROPORTIONAL ALGORITHM         #
############################################


def BiproportionalApportionment(VMatrix: WeightSeatMatrix) -> None:
    """
    Allocates seats following a biproportional scheme.

    >>> BiproportionalApportionment(WeightSeatMatrix_object)

    Returns
    -----------
    None. Updates WeightSeatMatrix_object.
    """
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
            return BiproportionalApportionment(VMatrix)
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
                return BiproportionalApportionment(VMatrix)
            else:
                return None
        (LabeledRows, LabeledColumns) = UpdateLabeled(
            VMatrix.IncrDecrMatrix, LabeledRows, LabeledColumns)
    return None

##############################################################
##                        EXAMPLES                          ##
##############################################################

# Let's try some of the functions and methods with the results of the 2018 and 2014 Brazilian general elections


# LOADING DATA
os.chdir(pathlib.Path(__file__).parent.resolve())
# 2018 Brazilian general election
Data_2018 = []
name_2018 = 'Data_2018.csv'
with open(name_2018, 'r') as dat:
    csv_reader = csv.reader(dat, delimiter=',')
    for row in csv_reader:
        Data_2018 += [list(row)]
for i in range(len(Data_2018)):
    Data_2018[i][0] = Data_2018[i][0].replace("\xa0", "")
    Data_2018[i][1] = int(Data_2018[i][1])
    Data_2018[i][2] = float(Data_2018[i][2])
    Data_2018[i][3] = int(Data_2018[i][3])
VectorCabeceras = ["Nombre Partido", "Nº Votos", "Porcentaje", "Escaños"]
# for data visualization
SortedAbbrevNames_2018 = ["Avante", "DC", "DEM", "MDB", "PCB", "PC do B", "PDT", "PHS", "NOVO", "PPS", "PP", "PPL", "PRTB", "PRB", "PRP", "PROS", "PSC",
                          "PSD", "PSL", "PSOL", "PSB", "PSTU", "PTB", "PTC", "PV", "PCO", "PMN", "PMB", "PR", "PSDB", "PT", "PATRI", "PODE", "REDE", "SOLID."]
Data_2018 = SortPerIndex(Data_2018, 0, False)
for i in range(len(Data_2018)):
    Data_2018[i][0] = SortedAbbrevNames_2018[i]

# 2014 Brazilian general election
Data_2014 = []
name_2014 = 'Data_2014.csv'
with open(name_2014, 'r') as dat:
    csv_reader = csv.reader(dat, delimiter=',')
    for row in csv_reader:
        Data_2014 += [list(row)]
for i in range(len(Data_2014)):
    Data_2014[i][0] = Data_2014[i][0].replace("\xa0", "")
    Data_2014[i][1] = int(Data_2014[i][1])
    Data_2014[i][2] = float(Data_2014[i][2])
    Data_2014[i][3] = int(Data_2014[i][3])
# for better data visualization
SortedAbbrevNames_2014 = ["DEM", "PCB", "PC do B", "PDT", "PEN", "PHS", "PPS", "PP", "PPL", "PRTB", "PRB", "PRP", "PROS", "PSC", "PSDC",
                          "PSD", "PSL", "PSOL", "PSB", "PSTU", "PTB", "PTC", "PTN", "PT do B", "PV", "PCO", "PMN", "PR", "PSDB", "PMDB", "PT", "SD"]
Data_2014 = SortPerIndex(Data_2014, 0, False)
for i in range(len(Data_2014)):
    Data_2014[i][0] = SortedAbbrevNames_2014[i]

# Federative units data 2018
zipFolder_2018 = zp.ZipFile('FU_Folder_2018.zip')
zipFolder_2018.extractall(path=None, members=None, pwd=None)
FU_names_2018 = zipFolder_2018.namelist()
FU_names_2018.sort()
FuData_2018 = []
for k in range(1, 28):
    DistrictData_2018 = []
    with open(FU_names_2018[k], 'r') as dat:
        csv_reader = csv.reader(dat, delimiter=',')
        for row in csv_reader:
            DistrictData_2018 += [list(row)]
    DistrictData_2018[1][0] = int(DistrictData_2018[1][0])
    for i in range(2, len(DistrictData_2018)):
        DistrictData_2018[i][0] = DistrictData_2018[i][0].replace("\xa0", "")
        if DistrictData_2018[i][0][0] == " ":
            DistrictData_2018[i][0] = DistrictData_2018[i][0][1::]
        DistrictData_2018[i][1] = int(DistrictData_2018[i][1])
        DistrictData_2018[i][2] = float(DistrictData_2018[i][2])
        DistrictData_2018[i][3] = int(DistrictData_2018[i][3])
    FuData_2018 += [DistrictData_2018]
TestData_2018 = cp.deepcopy(FuData_2018)

# Preparing data for WeightSeatMatrix class (2018)

Parties_2018 = []
for district in TestData_2018:
    for i in range(2, len(district)):
        if district[i][0] not in Parties_2018:
            Parties_2018 += [district[i][0]]
TestData_2018_F = []
for i in range(len(TestData_2018)):
    District = []
    District += [TestData_2018[i][0][0], TestData_2018[i][1][0]]
    AuxList = []
    VotedParties = []
    for j in range(2, len(TestData_2018[i])):
        AuxDict = {"PartyName": TestData_2018[i]
                   [j][0], "Votes": TestData_2018[i][j][1]}
        VotedParties += [TestData_2018[i][j][0]]
        AuxList += [AuxDict]
    for party in Parties_2018:
        if party not in VotedParties:
            AuxDict = {"PartyName": party, "Votes": 0}
            AuxList += [AuxDict]
    District += [AuxList]
    District[2] = sorted(District[2], key=lambda e: (e['PartyName']))
    TestData_2018_F += [District]
PartyNames_2018 = sorted(Parties_2018)
PartyNames_2018 = SortedAbbrevNames_2018
VoteMatrix_2018 = [[TestData_2018_F[i][2][j]["Votes"]
                    for j in range(len(PartyNames_2018))] for i in range(len(TestData_2018_F))]
DistrictNames_2018 = [TestData_2018_F[i][0]
                      for i in range(len(TestData_2018_F))]
DistrictSizes_2018 = [TestData_2018_F[i][1]
                      for i in range(len(TestData_2018_F))]

# Creation of WeightSeatMatrix object
TestDat_2018 = WeightSeatMatrix(VoteMatrix_2018, DistrictNames_2018,
                                PartyNames_2018, DistrictSizes_2018)


# Federative units data 2014
zipFolder_2014 = zp.ZipFile('FU_Folder_2014.zip')
zipFolder_2014.extractall(path=None, members=None, pwd=None)
FU_names_2014 = zipFolder_2014.namelist()
FU_names_2014.sort()
FuData_2014 = []
for k in range(1, 28):
    DistrictData_2014 = []
    with open(FU_names_2014[k], 'r') as dat:
        csv_reader = csv.reader(dat, delimiter=',')
        for row in csv_reader:
            DistrictData_2014 += [list(row)]
    DistrictData_2014[1][0] = int(DistrictData_2014[1][0])
    for i in range(2, len(DistrictData_2014)):
        DistrictData_2014[i][0] = DistrictData_2014[i][0].replace("\xa0", "")
        if DistrictData_2014[i][0][0] == " ":
            DistrictData_2014[i][0] = DistrictData_2014[i][0][1::]
        DistrictData_2014[i][1] = int(DistrictData_2014[i][1])
        DistrictData_2014[i][2] = float(DistrictData_2014[i][2])
        DistrictData_2014[i][3] = int(DistrictData_2014[i][3])
    FuData_2014 += [DistrictData_2014]
TestData_2014 = cp.deepcopy(FuData_2014)

# Preparing data for WeightSeatMatrix class (2014)

Parties_2014 = []
for district in TestData_2014:
    for i in range(2, len(district)):
        if district[i][0] not in Parties_2014:
            Parties_2014 += [district[i][0]]
TestData_2014_F = []
for i in range(len(TestData_2014)):
    District = []
    District += [TestData_2014[i][0][0], TestData_2014[i][1][0]]
    AuxList = []
    VotedParties = []
    for j in range(2, len(TestData_2014[i])):
        AuxDict = {"PartyName": TestData_2014[i]
                   [j][0], "Votes": TestData_2014[i][j][1]}
        VotedParties += [TestData_2014[i][j][0]]
        AuxList += [AuxDict]
    for party in Parties_2014:
        if party not in VotedParties:
            AuxDict = {"PartyName": party, "Votes": 0}
            AuxList += [AuxDict]
    District += [AuxList]
    District[2] = sorted(District[2], key=lambda e: (e['PartyName']))
    TestData_2014_F += [District]
PartyNames_2014 = sorted(Parties_2014)
PartyNames_2014 = SortedAbbrevNames_2014
VoteMatrix_2014 = [[TestData_2014_F[i][2][j]["Votes"]
                    for j in range(len(PartyNames_2014))] for i in range(len(TestData_2014_F))]
DistrictNames_2014 = [TestData_2014_F[i][0]
                      for i in range(len(TestData_2014_F))]
DistrictSizes_2014 = [TestData_2014_F[i][1]
                      for i in range(len(TestData_2014_F))]

# Creation of WeightSeatMatrix object
TestDat_2014 = WeightSeatMatrix(VoteMatrix_2014, DistrictNames_2014,
                                PartyNames_2014, DistrictSizes_2014)

#################################################
#               DATA MANIPULATION               #
#################################################

# GENERAL RESULTS 2018, 2014
"""
BiproportionalApportionment(TestDat_2018)
General2018 = [TestDat_2018.Parties] + [[Data_2018[i][1]
                                         for i in range(len(Data_2018))]] + [[Data_2018[i][3]
                                                                              for i in range(len(Data_2018))]] + [TestDat_2018.TotalPartySeats]
General2018 = SortPerIndex([list(i) for i in zip(*General2018)], 1)
# LatexClipboardDataDoubleColumn(
#    General2018, ["Partido", "Votos", "Esc.", "Esc.$^*$"])

BiproportionalApportionment(TestDat_2014)
General2014 = [TestDat_2014.Parties] + [[Data_2014[i][1]
                                         for i in range(len(Data_2014))]] + [[Data_2014[i][3]
                                                                              for i in range(len(Data_2014))]] + [TestDat_2014.TotalPartySeats]
General2014 = SortPerIndex([list(i) for i in zip(*General2014)], 1)
LatexClipboardDataDoubleColumn(
    General2014, ["Partido", "Votos", "Esc.", "Esc.$^*$"])
"""
# Desproportionality indices with electoral threshold
"""
# 2018
TotalSeatMatrix_2018 = []
n = 3
HeaderList_2018 = ["Orig."]
for i in [i*0.065/n for i in range(n+1)]:
    HeaderList_2018 += [str(round(100*i, 2))+"\%"]
    VoteMatrixAux = Threshold(VoteMatrix_2018, i)
    TestDatAux = WeightSeatMatrix(
        VoteMatrixAux, DistrictNames_2018, PartyNames_2018, DistrictSizes_2018)
    BiproportionalApportionment(TestDatAux)
    TotalSeatMatrix_2018 += [TestDatAux.TotalPartySeats]
TotalSeatMatrixPerc_2018 = TotalSeatMatrix_2018
for i in range(len(TotalSeatMatrix_2018)):
    TotalSeatMatrixPerc_2018[i] = [round(
        100*TotalSeatMatrix_2018[i][j]/513, 2) for j in range(len(TotalSeatMatrix_2018[i]))]
VotesPercentage_2018 = [Data_2018[i][2] for i in range(len(Data_2018))]

TotalSeatMatrixPerc_2018 = [[round(100*Data_2018[i][3]/513, 2)
                             for i in range(len(Data_2018))]] + TotalSeatMatrixPerc_2018
GalList_2018 = [LeastSquareIndex(VotesPercentage_2018, TotalSeatMatrixPerc_2018[i])[
    0] for i in range(len(TotalSeatMatrixPerc_2018))]
JeffList_2018 = [JeffersonIndex(VotesPercentage_2018, TotalSeatMatrixPerc_2018[i])[
    0] for i in range(len(TotalSeatMatrixPerc_2018))]
SaLaList_2018 = [SainteLagueIndex(VotesPercentage_2018, TotalSeatMatrixPerc_2018[i])[
    0] for i in range(len(TotalSeatMatrixPerc_2018))]

# 2014

TotalSeatMatrix_2014 = []
for i in [i*0.065/n for i in range(n+1)]:
    VoteMatrixAux = Threshold(VoteMatrix_2014, i)
    TestDatAux = WeightSeatMatrix(
        VoteMatrixAux, DistrictNames_2014, PartyNames_2014, DistrictSizes_2014)
    BiproportionalApportionment(TestDatAux)
    TotalSeatMatrix_2014 += [TestDatAux.TotalPartySeats]
# each list represents the results at a certain threshold (but the first one which is the original outcome)
TotalSeatMatrixPerc_2014 = TotalSeatMatrix_2014
for i in range(len(TotalSeatMatrix_2014)):
    TotalSeatMatrixPerc_2014[i] = [round(
        100*TotalSeatMatrix_2014[i][j]/513, 2) for j in range(len(TotalSeatMatrix_2014[i]))]
VotesPercentage_2014 = [Data_2014[i][2] for i in range(len(Data_2014))]

TotalSeatMatrixPerc_2014 = [[round(100*Data_2014[i][3]/513, 2)
                             for i in range(len(Data_2014))]] + TotalSeatMatrixPerc_2014
GalList_2014 = [LeastSquareIndex(VotesPercentage_2014, TotalSeatMatrixPerc_2014[i])[
    0] for i in range(len(TotalSeatMatrixPerc_2014))]
JeffList_2014 = [JeffersonIndex(VotesPercentage_2014, TotalSeatMatrixPerc_2014[i])[
    0] for i in range(len(TotalSeatMatrixPerc_2014))]
SaLaList_2014 = [SainteLagueIndex(VotesPercentage_2014, TotalSeatMatrixPerc_2014[i])[
    0] for i in range(len(TotalSeatMatrixPerc_2014))]


TotalIndexMatrix = [["BR-2018: I$_G$"]+GalList_2018, ["BR-2018: I$_J$"]+JeffList_2018,
                    ["BR-2018: I$_{S-L}$"]+SaLaList_2018, ["BR-2014: I$_G$"]+GalList_2014, ["BR-2014: I$_J$"]+JeffList_2014, ["BR-2014: I$_{S-L}$"]+SaLaList_2014]
HeaderList = ["Index/Barriers"] + HeaderList_2018
LatexClipboardData(TotalIndexMatrix, HeaderList)
"""
# seats obtained using the algorithm with 2014 election data
"""
BiproportionalApportionment(TestDat_2014)
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
#LatexClipboardData(FirstMatrix, FirstHeader)

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
#LatexClipboardData(SecondMatrix, SecondHeader)

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
#LatexClipboardData(ThirdMatrix, ThirdHeader)
"""

# Proportional-representation
Python script created as part of a Master's Thesis.

## Includes:
  - Webster and Jefferson methods implemented
  - Class for seat calculation and electoral weight manipulation
  - Tie and Transfer algorithm implemented via auxiliary functions also included
      - The algorithm uses d'Hondt divisor method for calculating the column restrictions of the seat matrix (seats obtained by each party)
      - It uses Sainte-Laguë divisor method for allocation of seats in columns and seat transfer
  - Some index of disproportionality implemented (Gallagher, Loosemore-Hanby, Jefferson (any %), Sainte-Laguë, Rae)
  - Some superfluous auxiliary functions for data visualization
  - documentation about the functions
  - Examples of usage with uploaded databases

## Databases loaded:
  - 2018 Brazilian general elections
    - Total seats obtained by party (Data_2018.csv)
    - Seats obtained by party in each electoral district (FU_Folder_2018.zip)
    - http://electionresources.org/br/deputies.php?election=2018
  - 2014 Brazilian general elections
    - Total seats obtained by party (Data_2014.csv)
    - Seats obtained by party in each electoral district (FU_Folder_2014.zip)
    - http://electionresources.org/br/deputies.php?election=2014&state=BR
## References:
<a id="1">[1]</a> Pukelsheim, F., 2017. Proportional representation. Springer International Publishing AG.

<a id="1">[2]</a> Gallagher, M., 1991. Proportionality, disproportionality and electoral systems. Electoral studies, 10(1), pp.33-51.

<a id="1">[3]</a> Karpov, A., 2008. Measurement of disproportionality in proportional representation systems. Mathematical and Computer Modelling, 48(9-10), pp.1421-1438.
    

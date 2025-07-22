"""This file contains some large CSS strings for styling widgets."""

FOCK = """
<style>
.table-fock {
  border-collapse: collapse;
  text-align: left;
}
.table-fock th {
  text-align: left;
}
</style>
"""

WIRES = """
<style>
.line {
  width: 100%;
  border-top: 2px solid;
}
.braket {
  display: grid;
  grid-template-rows: 30px 30px;
  grid-template-columns: 108px;
}
.square {
  height: 100px;
  border: 4px solid black;
  border-radius: 15px;
  grid-area: 1 / 2 / span 2 / span 1;
  align: center;
  text-align: center;
  line-height: 100px;
}
.modes-grid {
  width: 324px;
  display: grid;
  grid-template-columns: 108px 108px 108px;
  grid-template-rows: 50px 50px;
}
</style>
"""

TABLE = """
<style>
td {
  border: 1px solid;
}
th {
  border: 1px solid;
  background-color: #FBAB7E;
  font-size: 14px;
  text-align: center;
}
</style>
"""

STATE = """
<style>
  .state-table tr { display: block; float: left; }
  .state-table th { display: block; }
  .state-table td { display: block; }
  .state-table {
    margin: auto;
    border-collapse: collapse;
  }
</style>
"""

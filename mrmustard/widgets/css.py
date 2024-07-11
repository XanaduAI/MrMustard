"""This file contains some large CSS strings for styling widgets."""

FOCK_CSS = """
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

WIRES_CSS = """
<style>
.container-wires {{
  position: relative;
  width: 320px;
  margin-left: 20px;
  display: flex;
  padding-left: 20px;
}}
.square {{
  width: 100px;
  height: 100px;
  top: 10px;
  border: 4px solid black;
  flex: 1;
  border-radius: 15px;
}}

.in {{
  width: 100px;
  height: 108px;
  display: flex;
  flex-direction: column;
  flex: 1;
}}
.out {{
  height: 108px;
  display: flex;
  flex-direction: column;
  flex: 1;
}}
.bra {{
  position: static;
  flex: 1;
}}
.ket {{
  position: static;
  flex: 1;
}}

.grid-container {{
  display: flex;
  flex-direction: row;
  gap: 20px;
  width: 350px;
  margin-top: 40px;
  margin-left: 20px;
}}

.grid-item-wires {{
  width: 170px;
  height: 100px;
  display: flex;
  background-image: linear-gradient(to top, #f3e7e9 0%, #e3eeff 99%, #e3eeff 100%);
  border-radius: 15px;
  box-shadow: 3px 5px 10px black;
}}

.grid-item-type {{
  flex: 1;
  transform: translateY(30px);
  font-weight: bold;
  padding-left: 10px;
}}

.table-container-wires {{
  flex: 1.7;
  height: 90px;
  overflow-y: auto;
  margin: 5px;
  font-size: 12px;
}}

.table-wires {{
  text-align: center;
  margin-left: 30px;
  box-shadow: 3px 5px 10px black;
}}

.th-wires, td-wires {{
  border: 1px solid black;
}}

thead {{
  position: sticky;
  top: 0;
  background-color: #fff;
}}

.line {{
  position: relative;
  top: 50%;
  left: 0;
  width: 100%;
}}

.text-wires {{
  position: absolute;
  width: 100%;
  padding-left: 5px;
}}
.text-wires.type {{
  transform: translateY(25px);
}}
.text-wires.modes {{
  font-size: 13px;
  transform: translateY(2px);
}}
</style>
"""

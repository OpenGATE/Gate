/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDistributionFile.hh"

#include "GateDistributionFileMessenger.hh"
#include <math.h>
#include <CLHEP/Random/RandFlat.h>
#include "GateTools.hh"


GateDistributionFile::GateDistributionFile(const G4String& itsName)
  : GateVDistributionArray(itsName)
  , m_FileName()
  , m_column_for_X(0)
  , m_column_for_Y(1)
{
    m_messenger = new GateDistributionFileMessenger(this,itsName);
}
//___________________________________________________________________
GateDistributionFile::~GateDistributionFile()
{
}
//___________________________________________________________________
void GateDistributionFile::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent)
    	 <<"File : "         << m_FileName
         <<'{'    << m_column_for_X<<':'<<m_column_for_Y<<'}'
	 <<G4endl;
}
//___________________________________________________________________
void GateDistributionFile::Read()
{
    Clear();
    G4cout<<"OPENING FILE "<<m_FileName<<G4endl;
    std::ifstream f(m_FileName,std::ios::in);
    if (!f){
       G4cerr<<"[GateDistributionFile::Read] WARNING : File "<<m_FileName<<" can't be opened"<<G4endl;
       return;
    }
    G4String pattern;
    G4int k;
    if (m_column_for_X>=0){
	k = (m_column_for_X<m_column_for_Y) ? m_column_for_X : m_column_for_Y;
	for (G4int i=0;i<k;++i) pattern += "%*s ";
	pattern+="%f ";
    	k = abs(m_column_for_Y - m_column_for_X )-1;
    } else k = m_column_for_Y ;
    for (G4int i=0;i<k;++i) pattern += "%*s ";
    pattern+="%f";
    G4float x,y;
    G4float *addrFirst, *addrSecond;
    if (m_column_for_X<m_column_for_Y){
    	addrFirst  = &x;
	addrSecond = &y;
    } else {
    	addrFirst  = &y;
	addrSecond = &x;
    }

    while (!f.eof()) {
       char line[512];
       f.getline(line,511);
       if (f.good()) { // file line can be read
//               G4cout<<"VALUE READ IN FILE "<<m_FileName
// 	            <<"["<<GetArrayX().size()<<"]:"<<line<<G4endl;
          bool ok;
    	  if (m_column_for_X<0)
	    ok = (sscanf(line,pattern.c_str(),addrSecond)==1);
	  else
	    ok = (sscanf(line,pattern.c_str(),addrFirst,addrSecond)==2);
      	  if (ok){
    	    InsertPoint(x,y);
	  } else { //pattern problem
	    G4cerr<<"[GateDistributionFile::Read] WARNING : Line format unrecognised"
	          <<G4endl<<line<<G4endl;
	  }
       }
    }
    FillRepartition();
}

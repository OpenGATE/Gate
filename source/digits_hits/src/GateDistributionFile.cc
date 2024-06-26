/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
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
  //G4cout << GateTools::Indent(indent)
    //	 <<"File : "         << m_FileName
      //   <<'{'    << m_column_for_X<<':'<<m_column_for_Y<<'}'
	 //<< Gateendl;
}
//___________________________________________________________________
void GateDistributionFile::Read()
{
    Clear();
    G4cout<<"OPENING FILE "<<m_FileName<< Gateendl;
    std::ifstream f(m_FileName,std::ios::in);
    if (!f){
       G4cerr<<"[GateDistributionFile::Read] WARNING : File "<<m_FileName<<" can't be opened\n";
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

    char line[512];
    bool ok;

    while (!f.eof()) {
       f.getline(line,511);
       if (!f.good())  // file line can be read
	 continue;
//               G4cout<<"VALUE READ IN FILE "<<m_FileName
// 	            <<"["<<GetArrayX().size()<<"]:"<<line<< Gateendl;

	  //++x;
    	  if (m_column_for_X<0){
	    ok = (sscanf(line,pattern.c_str(),addrSecond)==1);
	    if(ok)
	      InsertPoint(y);
	  }
	  else {
	    ok = (sscanf(line,pattern.c_str(),addrFirst,addrSecond)==2);
	    if (ok)
	      InsertPoint(x,y);
	  }

      	  if (!ok){
	    G4cerr<<"[GateDistributionFile::Read] WARNING : Line format unrecognised (expected == '" << pattern << "' )"
	          << Gateendl<<line<< Gateendl;
	  }

    }
    FillRepartition();
}
void  GateDistributionFile::ReadMatrix2d() {
	   std::ifstream f(m_FileName);
	    if (!f.is_open()) {
	        G4cerr << "[GateDistributionFile::ReadMatrix2d] WARNING: File " << m_FileName << " can't be opened\n";
	        return;
	    }
	    std::string line;
	    std::vector<G4double> xValues;
	    bool isFirstLine = true;

	    while (std::getline(f, line)) {
	        std::stringstream iss(line);

	        if (isFirstLine) {
	            // Process x values from the first line
	            G4double x;

	            while (iss >> x) {
	                xValues.push_back(x);
	                if (iss.peek() == ',') iss.ignore();
	            }
	            isFirstLine = false;
	        } else {
	            // Process y and stddev values for subsequent lines
	            G4double y;
	            iss >> y;
	            if (iss.peek() == ',') iss.ignore();
	            // Read stddev values for each x value
	            for (size_t i = 0; i < xValues.size(); ++i) {
	                G4double stddev;
	                if (iss >> stddev) {
	                	// Insert the (x, y) -> stddev pair into the map using InsertPoint
	                   InsertPoint(xValues[i], y, stddev);
	                    if (iss.peek() == ',') iss.ignore();
	                }
	            }
	        }
	    }

	    f.close();
	    //G4cout << "Content of stddevMap:\n";
	    //for (const auto& entry : stddevMap) {
	       // std::pair<double, double> coordinates = entry.first;
	       // G4double stddev = entry.second;
	        //std::cout << "Coordinates: (" << coordinates.first << ", " << coordinates.second << "), stddev: " << stddev << std::endl;
	    //}

	    FillRepartition();
	}


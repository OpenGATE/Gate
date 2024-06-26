/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateVDistributionArray.hh"
#include "GateDistributionFile.hh"


#include <math.h>
#include <CLHEP/Random/RandFlat.h>
#include "GateTools.hh"
#include<set>

GateVDistributionArray::GateVDistributionArray(const G4String& itsName)
  : GateVDistribution(itsName)
  , m_arrayX()
  , m_arrayY()
  , m_minX(0), m_minY(0), m_maxX(0),m_maxY(0)
  , m_arrayRepartition()
  , m_factorX(1)
  , m_factorY(1)
  , m_autoStart(0)
{
}
//___________________________________________________________________
GateVDistributionArray::~GateVDistributionArray()
{
}
//___________________________________________________________________
G4double GateVDistributionArray::MinX() const
{
    return m_minX/m_factorX;
}
//___________________________________________________________________
G4double GateVDistributionArray::MinY() const
{
    return m_minY/m_factorY;
}
//___________________________________________________________________
G4double GateVDistributionArray::MaxX() const
{
    return m_maxX/m_factorX;
}
//___________________________________________________________________
G4double GateVDistributionArray::MaxY() const
{
    return m_maxY/m_factorY;
}
//___________________________________________________________________
void GateVDistributionArray::Clear()
{
    m_arrayX.clear();
    m_arrayX.clear();
    m_arrayRepartition.clear();
    m_minX=DBL_MAX;
    m_minY=DBL_MAX;
    m_maxX=-DBL_MAX;
    m_maxX=-DBL_MAX;
}
//_________________________________________________________________
G4double GateVDistributionArray::Value(G4double x) const
{

    if ( m_arrayX.empty() ) return 0;
    if ( m_arrayX.size() == 1 ) return m_arrayY[0]/m_factorY;
    x*=m_factorX;
    G4double x1,x2,y1,y2;
    G4int idx=FindIdxBefore(x,m_arrayX);

    if (idx==-1) {
    	x1 = m_arrayX[0]; y1 = m_arrayY[0];
    	x2 = m_arrayX[1]; y2 = m_arrayY[0];
    } else if (idx==(G4int)m_arrayX.size()-1){
    	x1 = m_arrayX[m_arrayX.size()-2]; y1 = m_arrayY[m_arrayX.size()-2];
       x2 = m_arrayX[m_arrayX.size()-1]; y2 = m_arrayY[m_arrayX.size()-1];
    } else {
    	x1 = m_arrayX[idx]  ; y1 = m_arrayY[idx];
    	x2 = m_arrayX[idx+1]; y2 = m_arrayY[idx+1];
    }
//G4cout << "m_arrayX:" << Gateendl;
  for (size_t i = 0; i < m_arrayX.size(); ++i) {
        G4cout << m_arrayX[i] << Gateendl;
    }

    //G4cout << "m_arrayY:" << Gateendl;
    for (size_t i = 0; i < m_arrayY.size(); ++i) {
        G4cout << m_arrayY[i] << Gateendl;
    }

     //return the linear interpolation
    return (y1 + (x-x1)*(y2-y1)/(x2-x1))/m_factorY;
}

//__________________________________________________________________
G4double GateVDistributionArray::RepartitionValue(G4double x) const
{
    if ( m_arrayX.empty() ) return 0;
    x*=m_factorX;

    G4double x1,x2,y1,y2;
    G4int idx=FindIdxBefore(x,m_arrayX);

    if (idx==-1) {
    	return 0;
    } else if (idx==(G4int)m_arrayX.size()-1){
    	return 1;
    } else {
    	x1 = m_arrayX[idx]  ; y1 = m_arrayRepartition[idx];
    	x2 = m_arrayX[idx+1]; y2 = m_arrayRepartition[idx+1];
    }

    // return the linear interpolation
    return (y1 + (x-x1)*(y2-y1)/(x2-x1))/m_factorY;
}
//___________________________________________________________________
G4double GateVDistributionArray::ShootRandom() const
{
    if (m_arrayRepartition.size()<2) return 0;
    G4double y = CLHEP::RandFlat::shoot();
    G4int idx = FindIdxBefore(y,m_arrayRepartition);
    if (idx<0) return m_minX;
    if (idx==(G4int)m_arrayY.size()-1) return m_maxX;
    G4double x1,x2,y1,y2;
    x1 = m_arrayX[idx]  ; y1 = m_arrayRepartition[idx];
    x2 = m_arrayX[idx+1]; y2 = m_arrayRepartition[idx+1];
    // return the linear interpolation
    return (x1 + (y-y1)*(x2-x1)/(y2-y1))/m_factorX;
}
//___________________________________________________________________
void GateVDistributionArray:: FillRepartition() {
	{
	    m_arrayRepartition.clear();
	    if (m_arrayX.size()<2) return;
	    m_arrayRepartition.resize(m_arrayX.size());
	    for (G4int i=0;i<(G4int)m_arrayX.size();++i){m_arrayRepartition[i]=0;}
	    for (G4int i=1;i<(G4int)m_arrayX.size();++i){
	    	G4double x1 = m_arrayX[i-1];
	    	G4double x2 = m_arrayX[i];
	    	G4double y1 = m_arrayY[i-1];
	    	G4double y2 = m_arrayY[i];
	    	m_arrayRepartition[i] = m_arrayRepartition[i-1] + (y1+y2)*(x2-x1)/2;
	    }
	    for (G4int i=0;i<(G4int)m_arrayX.size();++i){
	    	m_arrayRepartition[i]/=Integral();
	//	G4cout<<"Repartition["<<i<<"] : "<<m_arrayX[i]<<'\t'<<m_arrayRepartition[i]<< Gateendl;
	    }
	}
	}

//___________________________________________________________________
G4int GateVDistributionArray::FindIdxBefore(G4double x
                      ,const std::vector<G4double>& array) const
{
    if (!array.empty() && array[0]>x) return -1;
    for (G4int i=0;i<(G4int)array.size()-1;++i){
	if (array[i+1]>x) return i;
    }
    return (G4int)array.size()-1;
}
//___________________________________________________________________
void GateVDistributionArray::InsertPoint(G4double x,G4double y)
{
    G4int i = FindIdxBefore(x,m_arrayX)+1;
    if (i==(G4int)m_arrayX.size()){ //last element
	m_arrayX.push_back(x);
	m_arrayY.push_back(y);
    } else if ( (i>=0) && m_arrayX[i] == x ) { //element replacement
      m_arrayY[i] = y;
      //G4cerr<<"[GateDistributionArray::InsertPoint] WARNING : replacing value for "
  	   // <<x<< Gateendl;
    } else { //other cases
	m_arrayX.insert( m_arrayX.begin()+i , x);
	m_arrayY.insert( m_arrayY.begin()+i , y);
    }
    if (x<m_minX) m_minX=x;
    if (x>m_maxX) m_maxX=x;
    if (y<m_minY) m_minY=y;
    if (y>m_maxY) m_maxY=y;
}
void GateVDistributionArray::InsertPoint(G4double x, G4double y, G4double stddev) {
	    std::pair<G4double, G4double> coordinates = std::make_pair(x, y);
	    stddevMap[coordinates] = stddev;


	    if (x < m_minX) m_minX = x;
	    if (x > m_maxX) m_maxX = x;
	    if (y < m_minY) m_minY = y;
	    if (y > m_maxY) m_maxY = y;
	   //G4cout << "Inserted: (x=" << x << ", y=" << y << ") -> stddev=" << stddev << std::endl;

	}

//___________________________________________________________________
void GateVDistributionArray::InsertPoint(G4double y)
{
    if (GetSize()==0) InsertPoint(m_autoStart,y); else InsertPoint(m_maxX+1,y);
}


G4double GateVDistributionArray::Value2D(double x, double y) const {
    auto it = stddevMap.find(std::make_pair(x, y));
    if (it != stddevMap.end()) {
    	// If the point is found, return the standard deviation
        G4double stddev= it->second;
      // G4cout << "stddev for x=" << x << " and y=" << y << " is " << stddev<< G4endl;
        return stddev;
    } else {
       //G4cout << "stddev not found for x=" << x << " and y=" << y << ". Performing bilinear interpolation." << G4endl;
        G4double stddev = BilinearInterpolation(x, y);
        return stddev;
    }
}
G4double GateVDistributionArray::BilinearInterpolation(G4double x, G4double y) const {
	    auto lower_x_it = stddevMap.lower_bound(std::make_pair(x, -INFINITY));
	    auto upper_x_it = stddevMap.upper_bound(std::make_pair(x, INFINITY));

	    if (lower_x_it == stddevMap.end() || upper_x_it == stddevMap.end()) {
	        G4cerr << "Error: Unable to find adjacent points in x direction." << G4endl;
	        return 0.0;
	    }

	    // Adjust to make sure lower_x and upper_x are correct
	    G4double lower_x = (lower_x_it == stddevMap.begin()) ? lower_x_it->first.first : std::prev(lower_x_it)->first.first;
	    G4double upper_x = upper_x_it->first.first;

	    if (lower_x == upper_x) {
	        G4cerr << "Error: Lower and upper x values are the same." << G4endl;
	        return 0.0;
	    }

	    // Extract unique y values for these x coordinates
	    std::set<G4double> y_values;
	    for (const auto& entry : stddevMap) {
	        if (entry.first.first == lower_x || entry.first.first == upper_x) {
	            y_values.insert(entry.first.second);
	        }
	    }

	    if (y_values.empty()) {
	        G4cerr << "Error: No y values found for the given x values." << G4endl;
	        return 0.0;
	    }

	    auto lower_y_it = y_values.lower_bound(y);
	    auto upper_y_it = y_values.upper_bound(y);

	    if (lower_y_it == y_values.end() || upper_y_it == y_values.end()) {
	        G4cerr << "Error: Unable to find adjacent points in y direction." << G4endl;
	        return 0.0;
	    }

	    G4double lower_y = (lower_y_it == y_values.begin()) ? *lower_y_it : *std::prev(lower_y_it);
	    G4double upper_y = *upper_y_it;

	    if (lower_y == upper_y) {
	        G4cerr << "Error: Lower and upper y values are the same." << G4endl;
	        return 0.0;
	    }

	    // Retrieve the sigma values for the four surrounding points
	    G4double stddev11 = stddevMap.at(std::make_pair(lower_x, lower_y));
	    G4double stddev12 = stddevMap.at(std::make_pair(lower_x, upper_y));
	    G4double stddev21 = stddevMap.at(std::make_pair(upper_x, lower_y));
	    G4double stddev22 = stddevMap.at(std::make_pair(upper_x, upper_y));

	    //G4cout << "Interpolation points: (" << lower_x << ", " << lower_y << "), (" << upper_x << ", " << lower_y << "), ("
	          // << lower_x << ", " << upper_y << "), (" << upper_x << ", " << upper_y << ")" << G4endl;
	    //G4cout << "stddev values: " << stddev11 << ", " << stddev21 << ", " << stddev12 << ", " << stddev22 << G4endl;

	    G4double interpolatedValue = (stddev11 * (upper_x - x) * (upper_y - y) +
	    		stddev21 * (x - lower_x) * (upper_y - y) +
				stddev12 * (upper_x - x) * (y - lower_y) +
				stddev22 * (x - lower_x) * (y - lower_y)) /
	            ((upper_x - lower_x) * (upper_y - lower_y));

	    //G4cout << "Interpolated value: " << interpolatedValue << G4endl;
	    return interpolatedValue;
	}



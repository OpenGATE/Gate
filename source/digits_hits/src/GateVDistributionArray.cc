/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateVDistributionArray.hh"

#include <math.h>
#include <CLHEP/Random/RandFlat.h>
#include "GateTools.hh"


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
//___________________________________________________________________
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

    // return the linear interpolation
    return (y1 + (x-x1)*(y2-y1)/(x2-x1))/m_factorY;
}
//___________________________________________________________________
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
void GateVDistributionArray::FillRepartition()
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
      G4cerr<<"[GateDistributionArray::InsertPoint] WARNING : replacing value for "
  	    <<x<< Gateendl;
    } else { //other cases
	m_arrayX.insert( m_arrayX.begin()+i , x);
	m_arrayY.insert( m_arrayY.begin()+i , y);
    }
    if (x<m_minX) m_minX=x;
    if (x>m_maxX) m_maxX=x;
    if (y<m_minY) m_minY=y;
    if (y>m_maxY) m_maxY=y;
}
//___________________________________________________________________
void GateVDistributionArray::InsertPoint(G4double y)
{
    if (GetSize()==0) InsertPoint(m_autoStart,y); else InsertPoint(m_maxX+1,y);
}

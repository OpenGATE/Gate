/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateDistributionFile_h
#define GateDistributionFile_h 1

#include "GateVDistributionArray.hh"
#include "GateDistributionFileMessenger.hh"

class GateDistributionFileMessenger;
class GateDistributionFile : public GateVDistributionArray
{
  public:

    //! Constructor
    GateDistributionFile(const G4String& itsName);
    //! Destructor
    virtual ~GateDistributionFile() ;

    //! Setters
    inline void SetFileName(const G4String& fileName) {m_FileName=fileName;}
    inline void SetColumnX(G4int colX=0) {m_column_for_X=colX;}
    inline void SetColumnY(G4int colY=0) {m_column_for_Y=colY;}
    //! Getters
    inline G4String GetFileName() const {return m_FileName;}
    inline G4int GetColumnX() const {return m_column_for_X;}
    inline G4int GetColumnY() const {return m_column_for_Y;}

     void Read();
     void ReadMatrix2d();

    virtual void DescribeMyself(size_t indent);
  private:
    //! private members
    G4String m_FileName;
    G4int m_column_for_X;
    G4int m_column_for_Y;
    GateDistributionFileMessenger* m_messenger;
    std::map<std::pair<double, double>, double> stddevMap;

    std::vector<double> xValues;
        std::vector<double> yValues;

};


#endif

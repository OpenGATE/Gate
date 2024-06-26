/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateVDistributionArray_h
#define GateVDistributionArray_h 1

#include "GateVDistribution.hh"
#include <vector>

class GateVDistributionArray : public GateVDistribution
{
  public:

    //! Constructor
    GateVDistributionArray(const G4String& itsName);
    //! Destructor
    virtual ~GateVDistributionArray() ;

    //! Setters
    //! Getters
    inline void SetFactorX(G4double factor) {m_factorX=factor;}
    inline void SetFactorY(G4double factor) {m_factorY=factor;}


    G4double Integral() const {return m_arrayRepartition.back();}
    virtual G4double MinX() const;
    virtual G4double MinY() const;
    virtual G4double MaxX() const;
    virtual G4double MaxY() const;
    virtual G4double Value(G4double x) const;

    virtual G4double RepartitionValue(G4double x) const;
   G4double BilinearInterpolation(G4double x, G4double y) const ;
    // Returns a random number following the current distribution
    // should be optimised according to each distrbution type
    virtual G4double ShootRandom() const;
    size_t GetSize() const {return m_arrayX.size();}
    void Clear();
    void SetAutoStart(G4int start) {m_autoStart=start;}
    virtual G4double Value2D(G4double x, G4double y) const;
  protected:
     void InsertPoint(G4double x,G4double y, G4double sigma );
     void InsertPoint(G4double x, G4double y );
    void InsertPoint(G4double y );
    //! private function
    G4int FindIdxBefore(G4double x
            ,const std::vector<G4double>& array) const;


    void FillRepartition() ;

    std::vector<G4double>& GetArrayX() {return m_arrayX;}
    std::vector<G4double>& GetArrayY() {return m_arrayY;}

    std::vector<G4double>& GetArrayRepartition() {return m_arrayRepartition;}
  private:
    //! private members
    std::vector<G4double> m_arrayX;
    std::vector<G4double> m_arrayY;

    G4double m_minX;
    G4double m_minY;
    G4double m_maxX;
    G4double m_maxY;
    std::vector<G4double> m_arrayRepartition; //! repartition function calculus
    G4double m_factorX;
    G4double m_factorY;

    G4int m_autoStart;

    std::map<std::pair<double, double>, double> stddevMap;
    std::vector<double> xValues; // Stocker les valeurs x
    std::vector<double> yValues; // Stocker les valeurs y

};


#endif

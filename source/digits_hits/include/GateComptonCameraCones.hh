#ifndef GATECOMPTONCAMERACONES_HH
#define GATECOMPTONCAMERACONES_HH

#include "G4ThreeVector.hh"
class GateComptonCameraCones
{
public:
    GateComptonCameraCones();

    inline void SetEnergy1(G4double de)          { m_E1 = de; }
    inline G4double GetEnergy1() const                { return m_E1; }

    inline void SetEnergy2(G4double de)          { m_E2 = de; }
    inline G4double GetEnergy2() const                { return m_E2; }

    inline void SetEnergyR(G4double de)          { m_ER = de; }
    inline G4double GetEnergyR() const                { return m_ER; }


    inline void  SetPosition1(const G4ThreeVector& xyz)     { m_Pos1 = xyz; }
    inline const G4ThreeVector& GetPosition1() const             { return m_Pos1; }


    inline void  SetPosition2(const G4ThreeVector& xyz)     { m_Pos2 = xyz; }
    inline const G4ThreeVector& GetPosition2() const             { return m_Pos2; }


    inline void  SetPosition3(const G4ThreeVector& xyz)     { m_Pos3 = xyz; }
    inline const G4ThreeVector& GetPosition3() const             { return m_Pos3; }


    inline void  SetTrueFlag(const G4bool& flag)     { m_IsTrueCoind = flag; }
     inline G4bool GetTrueFlag() const                { return m_IsTrueCoind; }

     inline void  SetNumSingles(const G4int& num)     { m_nSingles = num; }
      inline G4bool GetNumSingles() const                { return m_nSingles; }

private:
  G4double m_E1;            // energy deposition of the first interaction
  G4double m_E2;            // energy deposition of the second interaction
  G4double m_ER;            // Total energy deposition except E1
  G4ThreeVector m_Pos1;  //
  G4ThreeVector  m_Pos2;//  Second interaction
  G4ThreeVector  m_Pos3;//  third interaction
  G4bool m_IsTrueCoind;
  G4int m_nSingles;

};

#endif // GATECOMPTONCAMERACONES_HH

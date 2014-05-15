/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GatePETVRTSettings_hh
#define GatePETVRTSettings_hh 1

/**
	@author Niklas Rehfeld <rehfeld@imnc.in2p3.fr>
*/

class G4Region;
#include "GateVFictitiousMap.hh"

typedef G4Region G4Envelope;
class GateFictitiousFastSimulationModel;
class GateTotalDiscreteProcess;
class GatePhantomSD;

namespace GatePETVRT {
		enum Approx {kStepwiseTrace,kVolumeTrace,kDiscreteKleinNishina, kNotSet};
}


class GatePETVRTSettings{
public:
    GatePETVRTSettings();
    ~GatePETVRTSettings();
	
	enum VerbosityLevel {NoVerbose,Verbose,DetailedVerbose};
    void RegisterEnvelope(G4Envelope*);
    void RegisterFictitiousMap(GateVFictitiousMap*, bool deleteWithThis);
    void RegisterFictitiousFastSimulationModel(GateFictitiousFastSimulationModel*, bool deleteWithThis);
    void RegisterTotalDiscreteProcess(GateTotalDiscreteProcess*, bool deleteWithThis);
    void RegisterPhantomSD(GatePhantomSD*, bool deleteWithThis);

    void SetFictitiousEnergy(double);
    void SetDiscardEnergy(double); //should be equal or below fictitious energy
    void SetApproximations(GatePETVRT::Approx);
	
    inline G4Envelope* GetEnvelope() const;
    inline GateVFictitiousMap* GetFictitiousMap() const;
    GateTotalDiscreteProcess* GetTotalDiscreteProcess() const;
    GateFictitiousFastSimulationModel* GetFictitiousFastSimulationModel();
    inline GatePETVRT::Approx GetApproximations() const;
    inline GatePhantomSD* GetPhantomSD() const;
    inline G4double GetFictitiousEnergy() const;
    inline G4double GetDiscardEnergy() const;
	inline void SetVerbosity(VerbosityLevel);
	inline VerbosityLevel GetVerbosity() const;

private:
    G4Envelope* pEnvelope;
    GateVFictitiousMap* pFictitiousMap;
    GateTotalDiscreteProcess* pTotalDiscreteProcess;
    bool m_nDeleteFictitiousMap, m_nDeleteFictitiousFastSimulationModel,m_nDeleteTotalDiscreteProcess,m_nDeletePhantomSD;
    GatePETVRT::Approx m_nApproximations;
    GateFictitiousFastSimulationModel* pFictitiousFastSimulationModel;
    GatePhantomSD* pPhantomSD;
    G4double m_nFictitiousEnergy;
    G4double m_nDiscardEnergy;
	VerbosityLevel m_nVerbosityLevel;
};

inline G4Envelope* GatePETVRTSettings::GetEnvelope() const
{
	return pEnvelope;
}

inline GateVFictitiousMap* GatePETVRTSettings::GetFictitiousMap() const
{
	return pFictitiousMap;
}

inline GatePETVRT::Approx GatePETVRTSettings::GetApproximations() const
{
	return m_nApproximations;
}

inline GatePhantomSD* GatePETVRTSettings::GetPhantomSD() const
{
	return pPhantomSD;
}

inline G4double GatePETVRTSettings::GetFictitiousEnergy() const
{
	return m_nFictitiousEnergy;
}
inline G4double GatePETVRTSettings::GetDiscardEnergy() const
{
	return m_nDiscardEnergy;
}
	inline void GatePETVRTSettings::SetVerbosity(GatePETVRTSettings::VerbosityLevel v)
{
	m_nVerbosityLevel=v;
}
	inline GatePETVRTSettings::VerbosityLevel GatePETVRTSettings::GetVerbosity() const
{
	return m_nVerbosityLevel;
}

#endif

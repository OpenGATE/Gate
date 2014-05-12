/*##########################################
#developed by Hermann Fuchs
#
#Christian Doppler Laboratory for Medical Radiation Research for Radiation Oncology
#Department of Radiation Oncology
#Medical University of Vienna
#
#and 
#
#Pierre Gueth
#CREATIS
#
#July 2012
##########################################
*/
#ifndef GATECERENKOVPROCESSMESSENGER_HH
#define GATECERENKOVPROCESSMESSENGER_HH

#include "GateVProcessMessenger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"

class GateVProcess;

class GateCerenkovMessenger: public GateVProcessMessenger
{
	public:
		GateCerenkovMessenger(GateVProcess* pb);
		virtual ~GateCerenkovMessenger();

		virtual void BuildCommands(G4String base);
		virtual void SetNewValue(G4UIcommand*, G4String);

		inline int GetMaxNumPhotonsPerStep() const { return maxNumPhotonsPerStep; }
		inline bool GetTrackSecondariesFirst() const { return trackSecondariesFirst; }
		inline double GetMaxBetaChangePerStep() const { return maxBetaChangePerStep; }
	protected:
		G4UIcmdWithAnInteger* pSetMaxNumPhotonsPerStep;
		G4UIcmdWithABool* pSetTrackSecondariesFirst;
		G4UIcmdWithADouble* pSetMaxBetaChangePerStep;

		int maxNumPhotonsPerStep;
		bool trackSecondariesFirst;
		double maxBetaChangePerStep;
};

#endif

/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATEFASTI124_HH
#define GATEFASTI124_HH

#include "G4Event.hh"

#include "GateVSource.hh"
#include "GateSimplifiedDecay.hh"

class GateFastI124
{
public:
	GateFastI124( GateVSource* );
	~GateFastI124();
	
	void InitializeFastI124();
	
	inline GateSimplifiedDecay* GetSimplifiedDecay()
	{ return m_simpleDecay; }
	
	void GenerateVertex( G4Event* );
	
private:
	GateVSource* m_source;
	GateSimplifiedDecay* m_simpleDecay;
	vector<psd>* m_particleVector;
	
};

#endif

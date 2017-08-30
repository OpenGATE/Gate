/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateRegionDoseStat.hh"

//-----------------------------------------------------------------------------
GateRegionDoseStat::GateRegionDoseStat(int i):id(i)
{
  sum_edep = 0.0;
  sum_squared_edep = 0.0;
  sum_temp_edep = 0.0;
  sum_dose = 0.0;
  sum_squared_dose = 0.0;
  sum_temp_dose = 0.0;
  last_event_id = -1;
  nb_hits = 0;
  nb_event_hits = 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::string GateRegionDoseStat::ToString()
{
  std::ostringstream oss;
  oss << id << " "
      << sum_edep << " " << sum_squared_edep << " " << sum_temp_edep
      << sum_dose << " " << sum_squared_dose << " " << sum_temp_dose
      << " last_event=" << last_event_id << " nb_hits=" << nb_hits
      << " nb_e_hits = " << nb_event_hits << " vol=" << volume;
  return oss.str();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateRegionDoseStat::Update(long event_id, double edep, double density)
{
  double dose = edep/density/volume/gray;
  if (edep == 0.0) dose = 0.0;
  ++nb_hits;
  if (event_id != last_event_id) {
    sum_squared_edep += (sum_temp_edep*sum_temp_edep);
    sum_edep += sum_temp_edep;
    sum_temp_edep = edep;
    sum_squared_dose += (sum_temp_dose*sum_temp_dose);
    sum_dose += sum_temp_dose;
    sum_temp_dose = dose;
    last_event_id = event_id;
    ++nb_event_hits;
  }
  else {
    sum_temp_edep += edep;
    sum_temp_dose += dose;
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Static
void GateRegionDoseStat::ComputeRegionVolumes(GateImageFloat & image,
                                              LabelToSingleRegionMapType & map)
{
  GateImageFloat::const_iterator pi = image.begin();
  while (pi != image.end()) {
    int label = *pi;
    auto it = map.find(label);
    if (it == map.end()) {
      auto region = std::make_shared<GateRegionDoseStat>(label);
      map[label] = region;
    }
    map[label]->volume += 1.0;
    ++pi;
  }
  for(auto &m:map) {
    m.second->volume *= image.GetVoxelVolume();
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Static
void GateRegionDoseStat::AddAggregatedRegion(LabelToSingleRegionMapType & map,
                                             LabelToSeveralRegionsMapType & regionsMap,
                                             std::vector<int> & labels)
{
  DD("AddAggregatedRegion");
  // Create region
  static int id=1000;
  auto region = std::make_shared<GateRegionDoseStat>(id); // FIXME ID !!
  region->volume = 0;
  // Update region and set in the map
  for(auto label:labels) {
    DD(label);
    region->volume += map[label]->volume;
    regionsMap[label].push_back(region);
  }
  DD(region->ToString());
  ++id;
}
//-----------------------------------------------------------------------------


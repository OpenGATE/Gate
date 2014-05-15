/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTools.hh"

//---------------------------------------------------------------------------
// Returns a string composed of a number of tabulations
G4String GateTools::Indent(size_t indent)
{
  G4String indentString;
  for (size_t i=0; i<indent ; i++)
    indentString += "\t";
  return indentString;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
// Checks if a tag can be found within a string.
// If the tag is found, returns everything before the tag.
// If the tag is not found, returns the whole string.
G4String GateTools::GetBaseName(const G4String& name,const G4String& tag)
{
  G4String::size_type tagPos = name.find(tag);

  if ( tagPos != G4String::npos )
    return name.substr(0,tagPos);
  else
    return name.substr();
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
// Looks for a GATE file: if the file can not be found in the current 
// directory, FindGateFile() looks for it in the directory $GATEHOME.
// The function returns either the full file path or "" (file not found).
G4String GateTools::FindGateFile(const G4String& fileName)
{

  G4String filePath;
  FILE* fp;

  filePath = fileName;
  fp = fopen(filePath,"r");

  if (!fp) {
    char* env = getenv("GATEHOME");

    if (env) {
      filePath = G4String(env) + "/" + fileName;
      fp = fopen(filePath,"r");
    }
  }

  if (fp) {
    fclose(fp);
    return filePath;
  } 
  else
    return "";

}
//---------------------------------------------------------------------------

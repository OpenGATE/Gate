/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include <string>
#include <utility>

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


//---------------------------------------------------------------------------
G4String GateTools::PathDirName(const G4String& path)
{ 
  G4String dir = "";

  std::size_t found = path.find_last_of("/\\");
  if (found < std::string::npos)
  {
    dir = path.substr(0, found + 1);
  }

  return dir;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
std::pair<G4String, G4String> GateTools::PathSplit(const G4String& path)
{
  G4String dir = "";
  G4String fileName = "";

  std::size_t found = path.find_last_of("/\\");
  if (found < std::string::npos)
  {
    dir = path.substr(0, found + 1);
    fileName = path.substr(found + 1, std::string::npos);
  } 
  else if (found == std::string::npos)
  {
    fileName = path;
  }

  return std::pair<G4String, G4String>(dir, fileName);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
std::pair<G4String, G4String> GateTools::PathSplitExt(const G4String& path)
{
  G4String root = "";
  G4String ext = "";

  std::pair<G4String, G4String> dirAndFileName = PathSplit(path);
  const G4String& dir = dirAndFileName.first;
  const G4String& fileName = dirAndFileName.second;

  std::size_t found = fileName.find_last_of(".");
  if (found > 0 && found < std::string::npos)
  {
    root = dir + fileName.substr(0, found);
    ext = fileName.substr(found, std::string::npos);
  }
  else
  {
    root = path;
  }

  return std::pair<G4String, G4String>(root, ext);
}
//---------------------------------------------------------------------------

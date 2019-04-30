//
// Created by mdupont on 04/04/19.
//

#include "GateFileExceptions.hh"
GateFileException::GateFileException(const std::string &arg) : runtime_error(arg)
{}
GateClosedFileException::GateClosedFileException(const std::string &arg) : GateFileException(arg)
{}
GateHeaderException::GateHeaderException(const std::string &arg) : GateFileException(arg)
{}
GateMissingHeaderException::GateMissingHeaderException(const std::string &arg) : GateHeaderException(arg)
{}
GateMalFormedHeaderException::GateMalFormedHeaderException(const std::string &arg) : GateHeaderException(arg)
{}
GateKeyAlreadyExistsException::GateKeyAlreadyExistsException(const std::string &arg) : GateHeaderException(arg)
{}
GateModeFileException::GateModeFileException(const std::string &arg) : GateFileException(arg)
{}

GateManagerException::GateManagerException(const std::string &arg) : GateFileException(arg)
{}

GateUnknownKindManagerException::GateUnknownKindManagerException(const std::string &arg) : GateManagerException(arg)
{}

GateKeyNotFoundInHeaderException::GateKeyNotFoundInHeaderException(const std::string &arg) : GateHeaderException(arg)
{}

GateTypeMismatchHeaderException::GateTypeMismatchHeaderException(const std::string &arg) : GateHeaderException(arg)
{}

GateNoTypeInHeaderException::GateNoTypeInHeaderException(const std::string &arg) : GateHeaderException(arg)
{}

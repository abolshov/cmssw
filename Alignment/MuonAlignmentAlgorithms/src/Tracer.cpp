#include "Alignment/MuonAlignmentAlgorithms/interface/Tracer.hpp"

Tracer& Tracer::instance()
{
    static Tracer t;
    return t;
}

void Tracer::write(std::string const& msg)
{
    m_file << msg << std::endl;
}

Tracer::Tracer()
{
    m_file.open("log.txt");
}

Tracer::~Tracer()
{
    m_file.close();
}
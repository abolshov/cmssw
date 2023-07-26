#ifndef TRACER_HPP
#define TRACER_HPP

#include <cassert>
#include <fstream>
#include <algorithm>
#include <iterator>

class Tracer
{
    public:
    static Tracer& instance();
    ~Tracer();

    Tracer(Tracer const& other) = delete;
    Tracer(Tracer&& other) = delete;
    Tracer& operator=(Tracer const& other) = delete;
    Tracer& operator=(Tracer&& other) = delete;

    template<typename T>
    Tracer& operator<<(T& data)
    {
        m_file << data;
        return *this;
    }

    template<typename T>
    Tracer& operator<<(T&& data)
    {
        m_file << data;
        return *this;
    }

    template<typename T>
    std::ostream_iterator<T> getOutIt(char const* delim)
    {
        return std::ostream_iterator<T>(m_file, delim);
    }

    void write(std::string const& msg);

    template<typename It>
    void copy(It begin, It end, char const* delim = " ")
    {
        assert(begin < end);
        using T = typename std::iterator_traits<It>::value_type;
        std::copy(begin, end, std::ostream_iterator<T>(m_file, delim));
    }

    private:
    std::ofstream m_file;

    Tracer();
};

#endif
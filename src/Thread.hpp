#include <cstdarg>
#include <process.h>
#include <Windows.h>

template<typename pStartPointType>
class Thread
{
    private:
        pStartPointType _pEntry;                        // Entry point function
        DWORD _ExitCode;                                // Exit Value
        HANDLE _Handle;                                 // Handle to the thread
        BOOL _fIsActive;                                // True if the thread is running
        BOOL _fIsAlive;                                 // Thread not destoryed
        DWORD _dwStackSize;                             // Stack Size of the Thread
        void** _prgStartArgs;                           // Arguments passed to function
        DWORD _dwThreadID;                              // Unique Thread ID

    public:
        void** GetStartArg()                            // Return the start arg
        {
            return _prgStartArgs;

        }

        DWORD GetTID()                                  // Return the thread id
        {
            return _dwThreadID;

        }

        BOOL IsActive()                                 // Return true if running
        {
            return _fIsActive;

        }

        BOOL IsAlive()                                  // True if Thread not Destroyed
        {
            return _fIsAlive;

        }

        void Join()                                     // Join the thread back
        {
            WaitForSingleObject(_Handle, INFINITE);

        }

        DWORD Start(DWORD cArgs = 0, ...)               // Start
        {
            if (!_prgStartArgs && cArgs > 0) {
                _prgStartArgs = new void*[cArgs];

                va_list rgArgs;

                va_start(rgArgs, cArgs);
                for (DWORD dwIndex = 0; dwIndex < cArgs; ++dwIndex) {
                    _prgStartArgs[dwIndex] = va_arg(rgArgs, void*);

                }

                va_end(rgArgs);

            }

            _Handle = (HANDLE)_beginthreadex(NULL, _dwStackSize, _pEntry, (void*)this, (unsigned)0, (unsigned*)&_dwThreadID);

            return (DWORD)_Handle;

        }

        void SetActive(BOOL fIsActive)                  // Set the thread as running
        {
            _fIsActive = fIsActive;

        }

        void SetAlive(BOOL fIsAlive)                    // Thread is Created
        {
            _fIsAlive = fIsAlive;

        }

        void SetEntryFunction(pStartPointType pEntry)   // Set Entry Function
        {
            _pEntry = pEntry;

        }

        void SetStackSize(DWORD dwStackSize)            // Set the Stack Size
        {
            _dwStackSize = dwStackSize;

        }

        void Sleep(DWORD dwMilliseconds)                // Sleep
        {
            ::Sleep(static_cast<DWORD>(dwMilliseconds));

        }

    public:
        Thread() 
        { 
            memset(this, 0, sizeof(*this));             // Zero out member variables

            _fIsAlive = 1; 
            
        }

        Thread(pStartPointType pEntryFunction, DWORD cArgs = 0, ...)
        {
            #if DEBUG
                memset(this, 0, sizeof(*this));         // If debug zero out memory
                                                        // before using
            #endif

            _dwThreadID = 0;
            _pEntry = pEntryFunction;
            _dwStackSize = 4092;                        // 4k Bytes Default stack size

            _pStartArg = new void*[cArgs];

            va_list rgArgs;

            va_start(rgArgs, cArgs);
            for (DWORD dwIndex = 0; dwIndex < cArgs; ++dwIndex) {
                _prgStartArgs[dwIndex] = va_arg(rgArgs, void*);

            }

            va_end(rgArgs);

        }

        ~Thread()
        {
            delete [] _prgStartArgs;

            #if DEBUG
                memset(this, 0, sizeof(*this));         // If debug cleanup our used memory

            #else
                _fIsAlive = 0;

            #endif
    
        }

};

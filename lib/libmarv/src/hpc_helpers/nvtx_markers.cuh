#ifndef HELPERS_NVTX_MARKERS_CUH
#define HELPERS_NVTX_MARKERS_CUH

/*
    Need to link with -lnvToolsExt to use this
*/

#ifdef __NVCC__

#ifndef NO_NVTOOLSEXT
#include <nvtx3/nvToolsExt.h>
#endif

#include <iostream>
#include <cstring>

namespace nvtx {

    inline
    void push_range(const std::string& name, int cid){
#ifndef NO_NVTOOLSEXT
        const uint32_t colors_[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff, 0xdeadbeef, 0x12345678, 0xabcdef42 };
        const int num_colors_ = sizeof(colors_)/sizeof(uint32_t);

        int color_id = cid;
        color_id = color_id%num_colors_;
        nvtxEventAttributes_t eventAttrib;
        std::memset(&eventAttrib, 0, sizeof(nvtxEventAttributes_t));
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = colors_[color_id];
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = name.c_str();
        nvtxRangePushEx(&eventAttrib);
        //std::cout << "push " << name << std::endl;
#endif
    }

    inline
    void pop_range(const std::string& /*name*/){
#ifndef NO_NVTOOLSEXT
        nvtxRangePop();
        //std::cerr << "pop " << name << std::endl;
#endif
    }

    inline
    void pop_range(){
#ifndef NO_NVTOOLSEXT
        nvtxRangePop();
        //std::cerr << "pop " << std::endl;
#endif
    }

    struct ScopedRange{
        ScopedRange() : ScopedRange("unnamed", 0){}
        ScopedRange(const std::string& name, int cid){
            push_range(name, cid);
        }
        ~ScopedRange(){
            pop_range();
        }
    };

} // namespace nvtx

#endif

#endif /* HELPERS_NVTX_MARKERS_CUH */

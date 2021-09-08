#define MEMORY_ARENA_MAX Gigabytes(4)
#define MEMORY_ARENA_COMMIT_SIZE Kilobytes(4)

struct MemoryArena {
	void* base;
	u64 max;
	u64 commit_pos;
	u64 allocation_pos;
};

struct MemoryInfo {
	MemoryArena permanent_arena;
	MemoryArena file_arena;
	MemoryArena calc_arena;
	MemoryArena temp_calc_arena;
};

global MemoryInfo g_MemoryInfo;

void *Win32_ReserveMemory(u64 size) {
	void *memory = VirtualAlloc(0,size,MEM_RESERVE, PAGE_NOACCESS);
	return memory;
}

void Win32_CommitMemory(void *memory, u64 size) {
	VirtualAlloc(memory, size, MEM_COMMIT, PAGE_READWRITE);
}

void Win32_ReleaseMemory(void *memory, u64 size) {
	VirtualFree(memory, size, MEM_DECOMMIT);
}

internal MemoryArena Memory_ArenaInitialise() {
    MemoryArena arena = {};
    arena.max = MEMORY_ARENA_MAX;
    arena.base = Win32_ReserveMemory(MEMORY_ARENA_MAX);
    return arena;
}

internal void * Memory_ArenaPush(MemoryArena *arena, u64 size) {
    void *memory = 0;
    
    if((arena->allocation_pos + size) > arena->commit_pos)
    {
        u64 required = size;
        required += MEMORY_ARENA_COMMIT_SIZE - 1;
        required -= required % MEMORY_ARENA_COMMIT_SIZE;
        
        Win32_CommitMemory((u8 *)arena->base + arena->commit_pos, required);
        arena->commit_pos += required;
    }
    memory = (u8 *)arena->base + arena->allocation_pos;
    arena->allocation_pos += size;
    return memory;
}

internal void Memory_ArenaPop(MemoryArena *arena, u64 size) {
    if(size > arena->allocation_pos)
    {
        size = arena->allocation_pos;
    }
	memset(arena->base, 0, arena->allocation_pos);
    arena->allocation_pos -= size;
}

internal void Memory_ArenaClear(MemoryArena *arena) {
    Memory_ArenaPop(arena, arena->allocation_pos);
}
internal void Memory_ArenaRelease(MemoryArena *arena) { 
    Win32_ReleaseMemory(arena->base, arena->commit_pos);
}
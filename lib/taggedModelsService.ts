const TAGGED_KEY = 'nei-viz-tagged-models';

export interface TaggedModel {
  id: string;
  userId: string;
  circuitConfigId: string;
  modelId: string;
  tagName: string;
  resnormValue: number;
  notes?: string;
  isInteresting: boolean;
  metadata?: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface CreateTaggedModelRequest {
  circuitConfigId: string;
  modelId: string;
  tagName: string;
  resnormValue: number;
  notes?: string;
  isInteresting?: boolean;
  metadata?: Record<string, unknown>;
}

function load(): TaggedModel[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(TAGGED_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function save(models: TaggedModel[]): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(TAGGED_KEY, JSON.stringify(models));
}

export class TaggedModelsService {
  static async getTaggedModelsForCircuit(
    _userId: string,
    circuitConfigId: string,
  ): Promise<TaggedModel[]> {
    return load().filter(m => m.circuitConfigId === circuitConfigId);
  }

  static async createTaggedModel(
    userId: string,
    req: CreateTaggedModelRequest,
  ): Promise<TaggedModel> {
    const models = load();
    const model: TaggedModel = {
      id: `tagged-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      userId,
      circuitConfigId: req.circuitConfigId,
      modelId: req.modelId,
      tagName: req.tagName,
      resnormValue: req.resnormValue,
      notes: req.notes,
      isInteresting: req.isInteresting ?? false,
      metadata: req.metadata,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    models.push(model);
    save(models);
    return model;
  }

  static async updateTaggedModel(
    modelId: string,
    updates: Partial<TaggedModel>,
  ): Promise<boolean> {
    const models = load();
    const idx = models.findIndex(m => m.id === modelId);
    if (idx === -1) return false;
    models[idx] = { ...models[idx], ...updates, updatedAt: new Date().toISOString() };
    save(models);
    return true;
  }

  static async deleteTaggedModel(modelId: string): Promise<boolean> {
    const models = load();
    const filtered = models.filter(m => m.id !== modelId);
    if (filtered.length === models.length) return false;
    save(filtered);
    return true;
  }

  static async deleteMultipleTaggedModels(modelIds: string[]): Promise<boolean> {
    const set = new Set(modelIds);
    save(load().filter(m => !set.has(m.id)));
    return true;
  }

  static async getTaggedModel(modelId: string): Promise<TaggedModel | null> {
    return load().find(m => m.id === modelId) ?? null;
  }

  static async getAllUserTaggedModels(_userId: string): Promise<TaggedModel[]> {
    return load();
  }

  static async getInterestingTaggedModels(_userId: string): Promise<TaggedModel[]> {
    return load().filter(m => m.isInteresting);
  }
}
